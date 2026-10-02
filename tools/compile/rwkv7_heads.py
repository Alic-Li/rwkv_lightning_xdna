# SPDX-License-Identifier: Apache-2.0
"""Eight parallel workers, four FP32 recurrent heads each, one dispatch/layer."""

import json
import os
import aie.iron as iron
from aie.iron import In, InOut, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_full import ROOT, KERNEL_ROOT, typ, external


@iron.jit
def design(state: InOut, vectors: In, y: Out):
    fn = external(
        "rwkv7_wkv_fp32",
        (
            "wkv7_fp32.cc"
            if os.environ.get("RWKV_XDNA_EXACT", "1") == "1"
            else "wkv7_vector_fp32.cc"
        ),
        [typ(4096), typ(384), typ(4160)],
    )
    ss, ps, outs, splits = [], [], [], []
    for i in range(8):
        ss.append(ObjectFifo(typ(4096), name=f"state_{i}", depth=1))
        ps.append(ObjectFifo(typ(384), name=f"p{i}", depth=1))
        outs.append(ObjectFifo(typ(4160), name=f"o{i}", depth=1))
        splits.append(outs[-1].cons().split([0, 4096], obj_types=[typ(4096), typ(64)]))

    def core(s, p, o, f):
        for _ in range_(4):
            sv, pv, ov = s.acquire(1), p.acquire(1), o.acquire(1)
            f(sv, pv, ov)
            s.release(1)
            p.release(1)
            o.release(1)

    workers = [
        Worker(core, [ss[i].cons(), ps[i].cons(), outs[i].prod(), fn], stack_size=12288)
        for i in range(8)
    ]

    def seq(s, p, y, hs, hp, hn, hy):
        for i in range(8):
            hs[i].fill(s, tap=TAP((131072,), i * 16384, [1, 1, 1, 16384], [0, 0, 0, 1]))
            hp[i].fill(p, tap=TAP((12288,), i * 256, [4, 6, 1, 64], [64, 2048, 0, 1]))
            hn[i].drain(
                s,
                tap=TAP((131072,), i * 16384, [1, 1, 1, 16384], [0, 0, 0, 1]),
                wait=True,
            )
            hy[i].drain(
                y, tap=TAP((2048,), i * 256, [1, 1, 1, 256], [0, 0, 0, 1]), wait=True
            )

    return Program(
        iron.get_current_device(),
        Runtime(
            seq,
            [
                typ(131072),
                typ(12288),
                typ(2048),
                [f.prod() for f in ss],
                [f.prod() for f in ps],
                [f[0].cons() for f in splits],
                [f[1].cons() for f in splits],
            ],
        ),
        workers=workers,
    ).resolve_program()


if __name__ == "__main__":
    p = KERNEL_ROOT / "array-decode"
    p.mkdir(parents=True, exist_ok=True)
    design.compile(p / "design.xclbin", p / "instructions.bin")
    (p / "config.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                dtype="float32",
                heads=32,
                head_size=64,
                exact_fp32=os.environ.get("RWKV_XDNA_EXACT", "1") == "1",
            )
        )
        + "\n"
    )
    print(p)

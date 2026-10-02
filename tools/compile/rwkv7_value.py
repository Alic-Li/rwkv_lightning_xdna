# SPDX-License-Identifier: Apache-2.0
"""Eight-way value residual, retaining the selected scalar arithmetic contract."""

import json
import os
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Worker, Runtime, Program
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_full import ROOT, KERNEL_ROOT, typ, external


@iron.jit
def design(aux: In, first: In, out: Out):
    aa = [ObjectFifo(typ(768), name=f"a{i}", depth=1) for i in range(8)]
    ff = [ObjectFifo(typ(256), name=f"f{i}", depth=1) for i in range(8)]
    oo = [ObjectFifo(typ(256), name=f"o{i}", depth=1) for i in range(8)]
    fn = external(
        "rwkv7_value_residual", "value_fp32.cc", [typ(768), typ(256), typ(256)]
    )

    def core(a, f, o, k):
        av, fv, ov = a.acquire(1), f.acquire(1), o.acquire(1)
        k(av, fv, ov)
        a.release(1)
        f.release(1)
        o.release(1)

    workers = [
        Worker(core, [aa[i].cons(), ff[i].cons(), oo[i].prod(), fn], stack_size=12288)
        for i in range(8)
    ]

    def seq(a, f, o, ha, hf, ho):
        for i in range(8):
            ha[i].fill(a, tap=TAP((6144,), i * 256, [1, 1, 3, 256], [0, 0, 2048, 1]))
            hf[i].fill(f, tap=TAP((2048,), i * 256, [1, 1, 1, 256], [0, 0, 0, 1]))
            ho[i].drain(
                o, tap=TAP((2048,), i * 256, [1, 1, 1, 256], [0, 0, 0, 1]), wait=True
            )

    return Program(
        iron.get_current_device(),
        Runtime(
            seq,
            [
                typ(6144),
                typ(2048),
                typ(2048),
                [f.prod() for f in aa],
                [f.prod() for f in ff],
                [f.cons() for f in oo],
            ],
        ),
        workers=workers,
    ).resolve_program()


if __name__ == "__main__":
    p = KERNEL_ROOT / "fused-value"
    p.mkdir(parents=True, exist_ok=True)
    design.compile(p / "design.xclbin", p / "instructions.bin")
    (p / "config.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                dtype="float32",
                channels=2048,
                exact_fp32=os.environ.get("RWKV_XDNA_EXACT", "1") == "1",
            )
        )
        + "\n"
    )

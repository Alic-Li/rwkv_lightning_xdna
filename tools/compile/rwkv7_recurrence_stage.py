# SPDX-License-Identifier: Apache-2.0
"""Fuse preparation, FP32 recurrent update and finishing in one dispatch."""

import json
import aie.iron as iron
from aie.iron import InOut, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_common import KERNEL_ROOT, typ, external


@iron.jit
def design(state: InOut, auxiliary: InOut):
    pre = external(
        "rwkv7_stage_prepare",
        "recurrence_stage_fp32.cc",
        [typ(1728), typ(1728)],
    )
    rec = pre.object_file.bind(
        "rwkv7_stage_recurrent", [typ(4096), typ(1728), typ(5824)]
    )
    post = pre.object_file.bind("rwkv7_stage_finish", [typ(1728), typ(1728)])
    ss = [ObjectFifo(typ(4096), name=f"state_{i}", depth=1) for i in range(8)]
    aa = [ObjectFifo(typ(1728), name=f"aux_{i}", depth=1) for i in range(8)]
    prepared = [ObjectFifo(typ(1728), name=f"prepared_{i}", depth=1) for i in range(8)]
    oo = [ObjectFifo(typ(5824), name=f"recout_{i}", depth=1) for i in range(8)]
    final = [ObjectFifo(typ(1728), name=f"final_{i}", depth=1) for i in range(8)]
    splits = [f.cons().split([0, 4096], obj_types=[typ(4096), typ(1728)]) for f in oo]

    def vector_stage(a, o, f):
        for _ in range_(4):
            av, ov = a.acquire(1), o.acquire(1)
            f(av, ov)
            a.release(1)
            o.release(1)

    def recurrent(s, a, o, f):
        for _ in range_(4):
            sv, av, ov = s.acquire(1), a.acquire(1), o.acquire(1)
            f(sv, av, ov)
            s.release(1)
            a.release(1)
            o.release(1)

    workers = []
    for i in range(8):
        workers += [
            Worker(
                vector_stage, [aa[i].cons(), prepared[i].prod(), pre], stack_size=12288
            ),
            Worker(
                recurrent,
                [ss[i].cons(), prepared[i].cons(), oo[i].prod(), rec],
                stack_size=12288,
            ),
            Worker(
                vector_stage,
                [splits[i][1].cons(), final[i].prod(), post],
                stack_size=12288,
            ),
        ]

    def seq(state, auxiliary, hs, ha, hn, ho):
        for i in range(8):
            st = TAP((131072,), i * 16384, [1, 1, 1, 16384], [0, 0, 0, 1])
            at = TAP((55296,), i * 256, [4, 27, 1, 64], [64, 2048, 0, 1])
            hs[i].fill(state, tap=st)
            ha[i].fill(auxiliary, tap=at)
            hn[i].drain(
                state,
                tap=TAP((131072,), i * 16384, [1, 1, 1, 16384], [0, 0, 0, 1]),
                wait=True,
            )
            ho[i].drain(
                auxiliary,
                tap=TAP((55296,), i * 256, [4, 27, 1, 64], [64, 2048, 0, 1]),
                wait=True,
            )

    return Program(
        iron.get_current_device(),
        Runtime(
            seq,
            [
                typ(131072),
                typ(55296),
                [f.prod() for f in ss],
                [f.prod() for f in aa],
                [f[0].cons() for f in splits],
                [f.cons() for f in final],
            ],
        ),
        workers=workers,
    ).resolve_program()


if __name__ == "__main__":
    path = KERNEL_ROOT / "fused-recurrence-stage"
    path.mkdir(parents=True, exist_ok=True)
    design.compile(path / "design.xclbin", path / "instructions.bin")
    (path / "config.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                dtype="float32",
                channels=2048,
                head_size=64,
                arena_vectors=27,
                exact_fp32=False,
            )
        )
        + "\n"
    )
    print(path, flush=True)

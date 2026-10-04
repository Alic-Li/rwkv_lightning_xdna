# SPDX-License-Identifier: Apache-2.0
"""Fuse preparation, FP32 recurrent update and finishing in one dispatch."""

import json
import numpy as np
import aie.iron as iron
from aie.iron import InOut, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_common import C, HEADS, KERNEL_ROOT, typ, external


def recurrence_program(with_value=False):
    lanes = 7 if with_value else 8
    head_counts = [HEADS // lanes + (i < HEADS % lanes) for i in range(lanes)]
    head_offsets = [sum(head_counts[:i]) for i in range(lanes)]
    aux_size = 1728
    input_size = 1920 if with_value else aux_size
    arena_size = HEADS * input_size
    pre = external(
        "rwkv7_value_stage_prepare" if with_value else "rwkv7_stage_prepare",
        "value_recurrence_fp32.cc" if with_value else "recurrence_prepare_fp32.cc",
        (
            [typ(input_size), typ(C), typ(aux_size), np.int32]
            if with_value
            else [typ(aux_size), typ(aux_size)]
        ),
        optimization="-Oz",
    )
    rec = external(
        "rwkv7_stage_recurrent",
        "recurrence_update_fp32.cc",
        [typ(4096), typ(aux_size), typ(4096 + aux_size)],
        optimization="-Os",
    )
    post = external(
        "rwkv7_stage_finish",
        "recurrence_finish_fp32.cc",
        [typ(aux_size), typ(aux_size)],
        optimization="-Os",
    )
    first_values = (
        ObjectFifo(typ(C), name="first_values", depth=1) if with_value else None
    )
    ss = [ObjectFifo(typ(4096), name=f"state_{i}", depth=1) for i in range(lanes)]
    aa = [ObjectFifo(typ(input_size), name=f"aux_{i}", depth=1) for i in range(lanes)]
    prepared = [
        ObjectFifo(typ(aux_size), name=f"prepared_{i}", depth=1) for i in range(lanes)
    ]
    oo = [
        ObjectFifo(typ(4096 + aux_size), name=f"recout_{i}", depth=1)
        for i in range(lanes)
    ]
    final = [
        ObjectFifo(typ(aux_size), name=f"final_{i}", depth=1) for i in range(lanes)
    ]
    splits = [
        f.cons().split([0, 4096], obj_types=[typ(4096), typ(aux_size)]) for f in oo
    ]

    def vector_stage(a, o, f, heads):
        for _ in range_(heads):
            av, ov = (a.acquire(1), o.acquire(1))
            f(av, ov)
            a.release(1)
            o.release(1)

    def recurrent(s, a, o, f, heads):
        for _ in range_(heads):
            sv, av, ov = (s.acquire(1), a.acquire(1), o.acquire(1))
            f(sv, av, ov)
            s.release(1)
            a.release(1)
            o.release(1)

    def prepare_value(a, first, o, f, heads, offset):
        fv = first.acquire(1)
        for h in range_(heads):
            av, ov = (a.acquire(1), o.acquire(1))
            f(av, fv, ov, h + offset)
            a.release(1)
            o.release(1)
        first.release(1)

    workers = []
    for i in range(lanes):
        workers += [
            (
                Worker(
                    prepare_value,
                    [
                        aa[i].cons(),
                        first_values.cons(),
                        prepared[i].prod(),
                        pre,
                        head_counts[i],
                        head_offsets[i],
                    ],
                    stack_size=12288,
                )
                if with_value
                else Worker(
                    vector_stage,
                    [aa[i].cons(), prepared[i].prod(), pre, head_counts[i]],
                    stack_size=12288,
                )
            ),
            Worker(
                recurrent,
                [ss[i].cons(), prepared[i].cons(), oo[i].prod(), rec, head_counts[i]],
                stack_size=12288,
            ),
            Worker(
                vector_stage,
                [splits[i][1].cons(), final[i].prod(), post, head_counts[i]],
                stack_size=12288,
            ),
        ]

    def schedule(state, auxiliary, hs, ha, hn, ho, first=None, hf=None):
        if with_value:
            hf.fill(first)
        for i in range(lanes):
            heads_per_lane = head_counts[i]
            head_offset = head_offsets[i]
            st = TAP(
                ((C * 64),),
                head_offset * 4096,
                [1, 1, 1, heads_per_lane * 4096],
                [0, 0, 0, 1],
            )
            at = TAP(
                (arena_size,),
                head_offset * 64,
                [heads_per_lane, input_size // 64, 1, 64],
                [64, C, 0, 1],
            )
            hs[i].fill(state, tap=st)
            ha[i].fill(auxiliary, tap=at)
            hn[i].drain(state, tap=st, wait=True)
            ho[i].drain(
                auxiliary,
                tap=TAP(
                    (arena_size,),
                    head_offset * 64,
                    [heads_per_lane, 27, 1, 64],
                    [64, C, 0, 1],
                ),
                wait=True,
            )

    def seq(state, auxiliary, hs, ha, hn, ho):
        schedule(state, auxiliary, hs, ha, hn, ho)

    def seq_value(state, auxiliary, first, hs, ha, hn, ho, hf):
        schedule(state, auxiliary, hs, ha, hn, ho, first, hf)

    program = Program(
        iron.get_current_device(),
        Runtime(
            seq_value if with_value else seq,
            [
                typ((C * 64)),
                typ(arena_size),
                *([typ(C)] if with_value else []),
                [f.prod() for f in ss],
                [f.prod() for f in aa],
                [f[0].cons() for f in splits],
                [f.cons() for f in final],
                *([first_values.prod()] if with_value else []),
            ],
        ),
        workers=workers,
    )
    return program.resolve_program()


@iron.jit
def design(state: InOut, auxiliary: InOut):
    return recurrence_program()


if __name__ == "__main__":
    path = KERNEL_ROOT / "fused-recurrence-stage"
    path.mkdir(parents=True, exist_ok=True)
    design.compile(path / "design.xclbin", path / "instructions.bin")
    (path / "config.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                dtype="float32",
                channels=C,
                head_size=64,
                arena_vectors=27,
                exact_fp32=False,
                kernel_optimization="-Os",
                prepare_optimization="-Oz",
            )
        )
        + "\n"
    )
    print(path, flush=True)

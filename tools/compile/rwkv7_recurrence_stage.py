# SPDX-License-Identifier: Apache-2.0
"""Fuse preparation, FP32 recurrent update and finishing in one dispatch."""

import json
import os
from aie.dialects.aie import event
import numpy as np
import aie.iron as iron
from aie.iron import InOut, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_common import KERNEL_ROOT, typ, external


TRACE_BYTES = int(os.environ.get("RWKV_XDNA_TRACE_BYTES", "0"))
if TRACE_BYTES < 0 or TRACE_BYTES > 64 * 1024 * 1024 or TRACE_BYTES % 4:
    raise ValueError("Invalid RWKV_XDNA_TRACE_BYTES")
OPTIMIZATION = os.environ.get("RWKV_XDNA_RECURRENCE_OPT", "-Os")
if OPTIMIZATION not in ("-Oz", "-Os", "-O2"):
    raise ValueError("Unsupported RWKV_XDNA_RECURRENCE_OPT")
PREPARE_TRACE_REGION = int(os.environ.get("RWKV_XDNA_PREPARE_TRACE_REGION", "0"))
if PREPARE_TRACE_REGION not in range(5) or (PREPARE_TRACE_REGION and not TRACE_BYTES):
    raise ValueError("RWKV_XDNA_PREPARE_TRACE_REGION must be 0..4 and requires trace bytes")


def recurrence_program(with_value=False):
    if PREPARE_TRACE_REGION == 4 and not with_value:
        raise ValueError("Value-region trace requires the fused value stage")
    # The eight-lane stage occupies all 16 shim S2MM channels. Its diagnostic
    # graph uses one lane. The fused value stage uses seven lanes: 14 state/aux
    # input streams plus one shared first-value broadcast fit the 16 MM2S channels.
    lanes = 7 if with_value else (1 if TRACE_BYTES else 8)
    head_counts = [5, 5, 5, 5, 4, 4, 4] if with_value else [32 // lanes] * lanes
    head_offsets = [sum(head_counts[:i]) for i in range(lanes)]
    aux_size = 1728
    input_size = 1920 if with_value else aux_size
    arena_size = 32 * input_size
    pre = external(
        "rwkv7_value_stage_prepare" if with_value else "rwkv7_stage_prepare",
        "value_recurrence_fp32.cc" if with_value else "recurrence_prepare_fp32.cc",
        [typ(input_size), typ(2048), typ(aux_size), np.int32] if with_value else [typ(aux_size), typ(aux_size)],
        # Preparation includes scalar underflow handling and FP64 statistics;
        # compile it for size to fit vector transcendental code on each tile.
        optimization="-Oz",
        extra_compile_flags=[f"-DRWKV_PREPARE_TRACE_REGION={PREPARE_TRACE_REGION}"],
    )
    rec = external(
        "rwkv7_stage_recurrent", "recurrence_update_fp32.cc",
        [typ(4096), typ(aux_size), typ(4096 + aux_size)],
        optimization=OPTIMIZATION,
    )
    post = external(
        "rwkv7_stage_finish", "recurrence_finish_fp32.cc",
        [typ(aux_size), typ(aux_size)], optimization=OPTIMIZATION,
    )
    first_values = ObjectFifo(typ(2048), name="first_values", depth=1) if with_value else None
    ss = [ObjectFifo(typ(4096), name=f"state_{i}", depth=1) for i in range(lanes)]
    aa = [ObjectFifo(typ(input_size), name=f"aux_{i}", depth=1) for i in range(lanes)]
    prepared = [ObjectFifo(typ(aux_size), name=f"prepared_{i}", depth=1) for i in range(lanes)]
    oo = [ObjectFifo(typ(4096 + aux_size), name=f"recout_{i}", depth=1) for i in range(lanes)]
    final = [ObjectFifo(typ(aux_size), name=f"final_{i}", depth=1) for i in range(lanes)]
    splits = [f.cons().split([0, 4096], obj_types=[typ(4096), typ(aux_size)]) for f in oo]

    def vector_stage(a, o, f, heads):
        for _ in range_(heads):
            av, ov = a.acquire(1), o.acquire(1)
            if TRACE_BYTES and not PREPARE_TRACE_REGION:
                event(0)
            f(av, ov)
            if TRACE_BYTES and not PREPARE_TRACE_REGION:
                event(1)
            a.release(1)
            o.release(1)

    def recurrent(s, a, o, f, heads):
        for _ in range_(heads):
            sv, av, ov = s.acquire(1), a.acquire(1), o.acquire(1)
            if TRACE_BYTES and not PREPARE_TRACE_REGION:
                event(0)
            f(sv, av, ov)
            if TRACE_BYTES and not PREPARE_TRACE_REGION:
                event(1)
            s.release(1)
            a.release(1)
            o.release(1)

    def prepare_value(a, first, o, f, heads, offset):
        fv = first.acquire(1)
        for h in range_(heads):
            av, ov = a.acquire(1), o.acquire(1)
            if TRACE_BYTES and not PREPARE_TRACE_REGION:
                event(0)
            f(av, fv, ov, h + offset)
            if TRACE_BYTES and not PREPARE_TRACE_REGION:
                event(1)
            a.release(1)
            o.release(1)
        first.release(1)

    workers = []
    for i in range(lanes):
        workers += [
            Worker(
                prepare_value,
                [aa[i].cons(), first_values.cons(), prepared[i].prod(), pre, head_counts[i], head_offsets[i]],
                stack_size=12288,
            ) if with_value else Worker(
                vector_stage, [aa[i].cons(), prepared[i].prod(), pre, head_counts[i]], stack_size=12288
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
                (131072,), head_offset * 4096,
                [1, 1, 1, heads_per_lane * 4096], [0, 0, 0, 1],
            )
            at = TAP(
                (arena_size,), head_offset * 64,
                [heads_per_lane, input_size // 64, 1, 64], [64, 2048, 0, 1],
            )
            hs[i].fill(state, tap=st)
            ha[i].fill(auxiliary, tap=at)
            hn[i].drain(
                state,
                tap=st,
                wait=True,
            )
            # The value inputs are immutable and already reside in DDR. Return
            # only the original 27-vector recurrence arena, avoiding their writeback.
            ho[i].drain(
                auxiliary,
                tap=TAP((arena_size,), head_offset * 64,
                        [heads_per_lane, 27, 1, 64], [64, 2048, 0, 1]),
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
                typ(131072),
                typ(arena_size),
                *([typ(2048)] if with_value else []),
                [f.prod() for f in ss],
                [f.prod() for f in aa],
                [f[0].cons() for f in splits],
                [f.cons() for f in final],
                *([first_values.prod()] if with_value else []),
            ],
        ),
        workers=workers,
    )
    if TRACE_BYTES:
        # Prepare, state update, finish; markers exclude FIFO acquire/release.
        program.enable_trace(
            trace_size=TRACE_BYTES, workers=workers[:1] if PREPARE_TRACE_REGION else workers[:3], egress_shim_col=7
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
                channels=2048,
                head_size=64,
                arena_vectors=27,
                exact_fp32=False,
                trace_buffer_bytes=TRACE_BYTES,
                kernel_optimization=OPTIMIZATION,
                prepare_optimization="-Oz",
                prepare_trace_region=PREPARE_TRACE_REGION,
            )
        )
        + "\n"
    )
    print(path, flush=True)

# SPDX-License-Identifier: Apache-2.0
"""Experimental single-token W1 -> activation256 -> W2 streaming FFN.

Four W1 and four W2 workers, unchanged per-output FP32 reduction order.
The W2 DMA visits existing row-major weight tiles in K-major order.
"""

import json
import os
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import In, InOut, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from aie.dialects.aie import event
from rwkv7_common import ROOT, KERNEL_ROOT, typ, external

INTERLEAVED = os.environ.get("RWKV_XDNA_FFN_INTERLEAVED", "0") == "1"
PACKED_K_MAJOR = os.environ.get("RWKV_XDNA_FFN_W2_PACKED_K_MAJOR", "0") == "1"

# Offline diagnostic build only. Tracing adds a trailing BO to the ABI.
TRACE_ACTIVATION = os.environ.get("RWKV_XDNA_TRACE_ACTIVATION", "0") == "1"
TRACE_BYTES = int(os.environ.get("RWKV_XDNA_TRACE_BYTES", "0"))
TRACE_PAIR = int(os.environ.get("RWKV_XDNA_TRACE_PAIR", "0"))
TRACE_ROLE = os.environ.get("RWKV_XDNA_TRACE_ROLE", "both")
if TRACE_ROLE not in ("both", "key", "value"):
    raise ValueError("RWKV_XDNA_TRACE_ROLE must be both, key or value")
if TRACE_PAIR not in range(4):
    raise ValueError("RWKV_XDNA_TRACE_PAIR must be 0..3")
if TRACE_BYTES < 0 or TRACE_BYTES % 4:
    raise ValueError("RWKV_XDNA_TRACE_BYTES must be nonnegative and word aligned")
if TRACE_ACTIVATION and not TRACE_BYTES:
    raise ValueError("RWKV_XDNA_TRACE_ACTIVATION requires RWKV_XDNA_TRACE_BYTES")


def channel_mix_program():
    wt = lambda n: np.ndarray[(n,), np.dtype[bfloat16]]
    tile = 4096
    weight_count, stripe = 8192 * tile, 1024 * tile
    matrix_stack = 12288
    key = external(
        "rwkv7_ffn_key_tile",
        "channel_mix_stream_bf16.cc",
        [typ(2048), wt(tile), typ(2048), np.int32, np.int32],
        optimization="-O3",
        stack_size=8192,
    )
    value = key.object_file.bind(
        "rwkv7_ffn_value_tile",
        [typ(1024 if INTERLEAVED else 256), wt(tile), typ(512), np.int32, np.int32],
    )
    zero = key.object_file.bind("rwkv7_ffn_zero2048", [typ(2048)])
    vz = key.object_file.bind("rwkv7_channel_zero512", [typ(512)])
    relu = key.object_file.bind(
        "rwkv7_ffn_activate_block256", [typ(2048), typ(256), np.int32]
    )
    copy256 = key.object_file.bind("rwkv7_ffn_copy256", [typ(256), typ(256)])
    add = key.object_file.bind("rwkv7_ffn_residual", [typ(4096), typ(4096)])
    norm_in = ObjectFifo(typ(8192), name="norm_input", depth=1)
    norm_join = norm_in.prod().join([0, 2048, 4096, 6144], obj_types=[typ(2048)] * 4)
    pair_norm = ObjectFifo(typ(4096), name="norm_pair", depth=1)
    coeff = ObjectFifo(typ(2048), name="coefficient", depth=1)
    norm = external("rwkv7_norm_pair", "norm_mix_fp32.cc", [typ(8192), typ(4096)])
    mix = external(
        "rwkv7_mix_pair", "mix_pair_fp32.cc", [typ(4096), typ(2048), typ(2048)]
    )
    fx = ObjectFifo(typ(2048), name="input", depth=1)
    w1 = [ObjectFifo(wt(tile), name=f"keyw{i}", depth=2) for i in range(4)]
    raw = ObjectFifo(typ(8192), name="raw", depth=1)
    kr = raw.prod().join([i * 2048 for i in range(4)], obj_types=[typ(2048)] * 4)
    if INTERLEAVED:
        # Worker j owns 256-row blocks j, j+4, ... . Each memory-tile join
        # produces the next contiguous 1024 activation channels for W2.
        act = ObjectFifo(typ(1024), name="active_stream", depth=2)
        ka = act.prod().join([i * 256 for i in range(4)], obj_types=[typ(256)] * 4)
        act_fanout = act
    else:
        act = ObjectFifo(typ(256), name="active_stream", depth=2)
        act_fanout = act.cons().forward(name="active_stream_fanout")
        groups = [
            ObjectFifo(typ(256), name=f"active_group{i}", depth=16) for i in range(2)
        ]
        ka = [ObjectFifo(typ(256), name=f"key_active{i}", depth=8) for i in range(4)]
    w2 = [ObjectFifo(wt(tile), name=f"valuew{i}", depth=2) for i in range(4)]
    pair = ObjectFifo(typ(4096), name="residual_pair", depth=1)
    joins = pair.prod().join(
        [0, 512, 1024, 1536, 2048],
        obj_types=[typ(512)] * 4 + [typ(2048)],
    )
    vo = joins[:4]
    final = ObjectFifo(typ(4096), name="result", depth=1)

    if TRACE_BYTES:
        trace_start = external(
            "rwkv7_stream_trace_start",
            "channel_mix_trace_markers.cc",
            [typ(1024 if INTERLEAVED else 256)],
        )
        trace_end = trace_start.object_file.bind("rwkv7_stream_trace_end", [typ(512)])

    def first(x, w, r, a, f, z, activation):
        xv, rv = x.acquire(1), r.acquire(1)
        if TRACE_BYTES:
            event(0)
        z(rv)
        for block in range_(8):
            for row in range_(16):
                for col in range_(8):
                    wv = w.acquire(1)
                    f(xv, wv, rv, block * 16 + row, col)
                    w.release(1)
            av = a.acquire(1)
            activation(rv, av, block)
            a.release(1)
        x.release(1)
        r.release(1)
        if TRACE_BYTES:
            event(1)

    def second(x, w, o, f, z, *markers):
        ov = o.acquire(1)
        z(ov)
        # Each output row still accumulates K tiles in original 0..31 order.
        # In trace builds delay START until first data arrives; cores can start
        # before the shim trace trigger, losing markers issued before that point.
        for group in range(2 if TRACE_BYTES else 1):
            chunks = 8 if INTERLEAVED else 32
            count = (1 if group == 0 else chunks - 1) if TRACE_BYTES else chunks
            for col in range_(count):
                xv = x.acquire(1)
                if TRACE_BYTES and group == 0:
                    markers[0](xv)
                for subcol in range_(4 if INTERLEAVED else 1):
                    for row in range_(32):
                        wv = w.acquire(1)
                        f(xv, wv, ov, row, subcol)
                        w.release(1)
                x.release(1)
        if TRACE_BYTES:
            markers[1](ov)
        o.release(1)

    def collect(x0, x1, o, f, count):
        for x in (x0, x1):
            for _ in range_(count):
                xv, ov = x.acquire(1), o.acquire(1)
                f(xv, ov)
                x.release(1)
                o.release(1)

    def finish(p, o, f):
        pv, ov = p.acquire(1), o.acquire(1)
        f(pv, ov)
        p.release(1)
        o.release(1)

    def normalize(p, o, f):
        pv, ov = p.acquire(1), o.acquire(1)
        f(pv, ov)
        p.release(1)
        o.release(1)

    def mixes(p, c, o, f):
        pv, cv, ov = p.acquire(1), c.acquire(1), o.acquire(1)
        f(pv, cv, ov)
        p.release(1)
        c.release(1)
        o.release(1)

    workers = [
        Worker(
            normalize,
            [norm_in.cons(), pair_norm.prod(), norm],
            stack_size=12288,
        ),
        Worker(
            mixes,
            [pair_norm.cons(), coeff.cons(), fx.prod(), mix],
            stack_size=12288,
        ),
    ]
    workers += [
        Worker(
            first,
            [
                fx.cons(),
                w1[i].cons(),
                kr[i].prod(),
                ka[i].prod(),
                key,
                zero,
                relu,
            ],
            stack_size=matrix_stack,
        )
        for i in range(4)
    ]
    workers += [
        Worker(
            second,
            [
                act_fanout.cons(),
                w2[i].cons(),
                vo[i].prod(),
                value,
                vz,
                *([trace_start, trace_end] if TRACE_BYTES else []),
            ],
            stack_size=matrix_stack,
        )
        for i in range(4)
    ]
    workers += [Worker(finish, [pair.cons(), final.prod(), add], stack_size=12288)]

    if not INTERLEAVED:
        workers += [
            Worker(
                collect,
                [
                    ka[2 * i].cons(depth=1),
                    ka[2 * i + 1].cons(depth=1),
                    groups[i].prod(),
                    copy256,
                    8,
                ],
                stack_size=1024,
            )
            for i in range(2)
        ]
        workers += [
            Worker(
                collect,
                [
                    groups[0].cons(depth=1),
                    groups[1].cons(depth=1),
                    act.prod(),
                    copy256,
                    16,
                ],
                stack_size=1024,
            )
        ]

    def seq(
        x,
        parameters,
        w,
        diag,
        out,
        hx,
        hw,
        hb,
        hold,
        hc,
        hn,
        hm,
        hw1,
        hw2,
        hr,
        ha,
        hres,
        ho,
    ):
        hx.fill(x)
        hw.fill(parameters, tap=TAP((6144,), 0, [1, 1, 1, 2048], [0, 0, 0, 1]))
        hb.fill(parameters, tap=TAP((6144,), 2048, [1, 1, 1, 2048], [0, 0, 0, 1]))
        hold.fill(diag, tap=TAP((22528,), 0, [1, 1, 1, 2048], [0, 0, 0, 1]))
        hc.fill(parameters, tap=TAP((6144,), 4096, [1, 1, 1, 2048], [0, 0, 0, 1]))
        hn.drain(diag, tap=TAP((22528,), 0, [1, 1, 1, 4096], [0, 0, 0, 1]), wait=True)
        hm.drain(
            diag, tap=TAP((22528,), 4096, [1, 1, 1, 2048], [0, 0, 0, 1]), wait=True
        )
        for i in range(4):
            hw1[i].fill(
                w,
                tap=(
                    TAP(
                        (weight_count,),
                        i * 524288,
                        [1, 8, 1, 524288],
                        [0, 2097152, 0, 1],
                    )
                    if INTERLEAVED
                    else TAP(
                        (weight_count,), i * stripe, [1, 1, 1, stripe], [0, 0, 0, 1]
                    )
                ),
            )
        for i in range(4):
            hw2[i].fill(
                w,
                tap=TAP(
                    (weight_count,),
                    weight_count // 2 + i * stripe,
                    [1, 1, 1, stripe] if PACKED_K_MAJOR else [1, 32, 32, tile],
                    [0, 0, 0, 1] if PACKED_K_MAJOR else [0, tile, 32 * tile, 1],
                ),
            )
        hr.drain(
            diag,
            tap=(
                TAP((22528,), 6144, [1, 4, 8, 256], [0, 256, 1024, 1])
                if INTERLEAVED
                else TAP((22528,), 6144, [1, 1, 1, 8192], [0, 0, 0, 1])
            ),
            wait=True,
        )
        ha.drain(
            diag, tap=TAP((22528,), 14336, [1, 1, 1, 8192], [0, 0, 0, 1]), wait=True
        )
        hres.fill(x)
        ho.drain(out, wait=True)

    program = Program(
        iron.get_current_device(),
        Runtime(
            seq,
            [
                typ(2048),
                typ(6144),
                wt(weight_count),
                typ(22528),
                typ(4096),
                *[f.prod() for f in norm_join],
                coeff.prod(),
                pair_norm.cons(),
                fx.cons(),
                [f.prod() for f in w1],
                [f.prod() for f in w2],
                raw.cons(),
                act_fanout.cons(),
                joins[4].prod(),
                final.cons(),
            ],
        ),
        workers=workers,
    )
    if TRACE_BYTES:
        # One key core and one value core: vector issue, memory/stream/lock
        # stalls and DMA port activity, using the compiler's default events.
        program.enable_trace(
            trace_size=TRACE_BYTES,
            workers=(
                [workers[2 + TRACE_PAIR]]
                if TRACE_ROLE == "key"
                else (
                    [workers[6 + TRACE_PAIR]]
                    if TRACE_ROLE == "value"
                    else [workers[2 + TRACE_PAIR], workers[6 + TRACE_PAIR]]
                )
            ),
            egress_shim_col=7,
        )
    return program.resolve_program()


@iron.jit
def design(x: In, parameters: In, weights: In, diagnostic: InOut, result: Out):
    return channel_mix_program()


if __name__ == "__main__":
    if (
        not os.environ.get("RWKV_XDNA_KERNEL_DIR")
        or KERNEL_ROOT.resolve() == (ROOT / "build/kernels/rwkv7-bf16").resolve()
    ):
        raise ValueError("Choose an isolated experimental RWKV_XDNA_KERNEL_DIR")
    if TRACE_ACTIVATION:
        raise ValueError(
            "Stream trace marks whole W1 / W2 with input waits; activation-only trace unsupported"
        )
    path = KERNEL_ROOT / "bf16-channel-mix"
    path.mkdir(parents=True, exist_ok=True)
    design.compile(path / "design.xclbin", path / "instructions.bin")
    (path / "config.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                dtype="bfloat16",
                channels=2048,
                hidden=8192,
                key_cores=4,
                value_cores=4,
                experimental="stream_activation256_k_major_w2",
                weight_layout="w2_k_major_4" if PACKED_K_MAJOR else "row_major",
                w1_output_partition=(
                    "block_cyclic_256" if INTERLEAVED else "contiguous_2048"
                ),
                collector_cores=0 if INTERLEAVED else 3,
                trace_value_region="first_activation_to_finish_including_later_activation_wait",
                exact_fp32=False,
                trace_buffer_bytes=TRACE_BYTES,
                trace_pair=TRACE_PAIR,
                trace_role=TRACE_ROLE,
                trace_key_region=(
                    "activation" if TRACE_ACTIVATION else "matrix_and_activation"
                ),
            )
        )
        + "\n"
    )
    print(path, flush=True)

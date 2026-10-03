# SPDX-License-Identifier: Apache-2.0
"""Native LayerNorm/shift/mix plus BF16 FFN with on-chip intermediates."""

import json
import os
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import In, InOut, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.helpers.taplib import TensorAccessPattern as TAP
from aie.dialects.aie import event
from rwkv7_common import KERNEL_ROOT, typ, external

# Offline diagnostic build only. Tracing adds a trailing BO to the ABI.
TRACE_ACTIVATION = os.environ.get("RWKV_XDNA_TRACE_ACTIVATION", "0") == "1"
TRACE_BYTES = int(os.environ.get("RWKV_XDNA_TRACE_BYTES", "0"))
if TRACE_BYTES < 0 or TRACE_BYTES % 4:
    raise ValueError("RWKV_XDNA_TRACE_BYTES must be nonnegative and word aligned")
if TRACE_ACTIVATION and not TRACE_BYTES:
    raise ValueError("RWKV_XDNA_TRACE_ACTIVATION requires RWKV_XDNA_TRACE_BYTES")


def channel_mix_program(quantized=False):
    # Compilation-time specialization; both programs have the same activation
    # ABI, but distinct weight types, tile strides and artifact contracts.
    place = lambda col, row: Tile(col, row) if quantized else None
    bt = lambda n: np.ndarray[(n,), np.dtype[bfloat16]]
    activation_type = bt if quantized else typ
    wt = lambda n: np.ndarray[(n,), np.dtype[np.uint8 if quantized else bfloat16]]
    tile = 4160 if quantized else 4096
    weight_count, stripe = 8192 * tile, 1024 * tile
    matrix_stack = 20480 if quantized else 12288
    key = external(
        "rwkv7_ffn_key_tile",
        "channel_mix_int8.cc" if quantized else "channel_mix_bf16.cc",
        [activation_type(2048), wt(tile), typ(2048), np.int32, np.int32],
        optimization="-O3",
        stack_size=16384 if quantized else 8192,
    )
    value = key.object_file.bind(
        "rwkv7_ffn_value_tile",
        [activation_type(8192), wt(tile), typ(512), np.int32, np.int32],
    )
    zero = key.object_file.bind("rwkv7_ffn_zero2048", [typ(2048)])
    vz = key.object_file.bind("rwkv7_channel_zero512", [typ(512)])
    relu = key.object_file.bind("rwkv7_ffn_activate2048", [typ(2048), typ(2048)])
    add = key.object_file.bind("rwkv7_ffn_residual", [typ(4096), typ(4096)])
    norm_in = ObjectFifo(typ(8192), name="norm_input", depth=1)
    norm_join = norm_in.prod().join(
        [0, 2048, 4096, 6144], tile=place(0, 1), obj_types=[typ(2048)] * 4
    )
    pair_norm = ObjectFifo(typ(4096), name="norm_pair", depth=1)
    coeff = ObjectFifo(typ(2048), name="coefficient", depth=1)
    norm = external("rwkv7_norm_pair", "norm_mix_fp32.cc", [typ(8192), typ(4096)])
    mix = external(
        "rwkv7_mix_pair", "mix_pair_fp32.cc", [typ(4096), typ(2048), typ(2048)]
    )
    fx = ObjectFifo(typ(2048), name="input", depth=1)
    if quantized:
        # Round each activation once, then multicast through memory tiles. Explicit
        # placement keeps the eight matrix workers off the converter/normalizer row
        # and avoids exhausting compute-tile peer-DMA channels during fanout.
        fx_bf16 = ObjectFifo(bt(2048), name="input_bf16", depth=1)
        fx_half = fx_bf16.cons().forward(tile=place(1, 1), name="input_bf16_fanout")
        act_bf16 = ObjectFifo(bt(8192), name="active_bf16", depth=1)
        act_half = act_bf16.cons().forward(tile=place(5, 1), name="active_bf16_fanout")
        convert = external(
            "rwkv7_channel_convert2048",
            "channel_mix_convert_bf16.cc",
            [typ(2048), bt(2048)],
            optimization="-O3",
        )
        convert_act = convert.object_file.bind(
            "rwkv7_channel_convert8192", [typ(8192), bt(8192)]
        )
    w1 = [ObjectFifo(wt(tile), name=f"keyw{i}", depth=2) for i in range(4)]
    raw = ObjectFifo(typ(8192), name="raw", depth=1)
    act = ObjectFifo(typ(8192), name="active", depth=1)
    kr = raw.prod().join(
        [i * 2048 for i in range(4)], tile=place(2, 1), obj_types=[typ(2048)] * 4
    )
    ka = act.prod().join(
        [i * 2048 for i in range(4)], tile=place(3, 1), obj_types=[typ(2048)] * 4
    )
    w2 = [ObjectFifo(wt(tile), name=f"valuew{i}", depth=2) for i in range(4)]
    pair = ObjectFifo(typ(4096), name="residual_pair", depth=1)
    joins = pair.prod().join(
        [0, 512, 1024, 1536, 2048],
        tile=place(7, 1),
        obj_types=[typ(512)] * 4 + [typ(2048)],
    )
    vo = joins[:4]
    final = ObjectFifo(typ(4096), name="result", depth=1)

    def first(x, w, r, a, f, z, activation):
        xv, rv, av = x.acquire(1), r.acquire(1), a.acquire(1)
        if TRACE_BYTES and not TRACE_ACTIVATION:
            event(0)
        z(rv)
        for row in range_(128):
            for col in range_(8):
                wv = w.acquire(1)
                f(xv, wv, rv, row, col)
                w.release(1)
        if TRACE_BYTES and TRACE_ACTIVATION:
            event(0)
        activation(rv, av)
        x.release(1)
        r.release(1)
        a.release(1)
        if TRACE_BYTES:
            event(1)

    def second(x, w, o, f, z):
        xv, ov = x.acquire(1), o.acquire(1)
        if TRACE_BYTES:
            event(0)
        z(ov)
        for row in range_(32):
            for col in range_(32):
                wv = w.acquire(1)
                f(xv, wv, ov, row, col)
                w.release(1)
        x.release(1)
        o.release(1)
        if TRACE_BYTES:
            event(1)

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
            tile=place(0, 2),
            stack_size=12288,
        ),
        Worker(
            mixes,
            [pair_norm.cons(), coeff.cons(), fx.prod(), mix],
            tile=place(1, 2),
            stack_size=12288,
        ),
    ]
    workers += [
        Worker(
            first,
            [
                (fx_half if quantized else fx).cons(),
                w1[i].cons(),
                kr[i].prod(),
                ka[i].prod(),
                key,
                zero,
                relu,
            ],
            tile=place(i, 3),
            stack_size=matrix_stack,
        )
        for i in range(4)
    ]
    workers += [
        Worker(
            second,
            [
                (act_half if quantized else act).cons(),
                w2[i].cons(),
                vo[i].prod(),
                value,
                vz,
            ],
            tile=place(i + 4, 3),
            stack_size=matrix_stack,
        )
        for i in range(4)
    ]
    workers += [
        Worker(
            finish, [pair.cons(), final.prod(), add], tile=place(7, 2), stack_size=12288
        )
    ]

    if quantized:
        workers += [
            Worker(
                normalize,
                [fx.cons(), fx_bf16.prod(), convert],
                tile=place(2, 2),
                stack_size=12288,
            ),
            Worker(
                normalize,
                [act.cons(), act_bf16.prod(), convert_act],
                tile=place(3, 2),
                stack_size=12288,
            ),
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
                w, tap=TAP((weight_count,), i * stripe, [1, 1, 1, stripe], [0, 0, 0, 1])
            )
        for i in range(4):
            hw2[i].fill(
                w,
                tap=TAP(
                    (weight_count,),
                    weight_count // 2 + i * stripe,
                    [1, 1, 1, stripe],
                    [0, 0, 0, 1],
                ),
            )
        hr.drain(
            diag, tap=TAP((22528,), 6144, [1, 1, 1, 8192], [0, 0, 0, 1]), wait=True
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
                act.cons(),
                joins[4].prod(),
                final.cons(),
            ],
        ),
        workers=workers,
    )
    if TRACE_BYTES:
        # One key core and one value core: vector issue, memory/stream/lock
        # stalls and DMA port activity, using the compiler's default events.
        program.enable_trace(trace_size=TRACE_BYTES, workers=[workers[2], workers[6]])
    return program.resolve_program()


@iron.jit
def design(x: In, parameters: In, weights: In, diagnostic: InOut, result: Out):
    return channel_mix_program()


if __name__ == "__main__":
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
                exact_fp32=False,
                trace_buffer_bytes=TRACE_BYTES,
                trace_key_region=(
                    "activation" if TRACE_ACTIVATION else "matrix_and_activation"
                ),
            )
        )
        + "\n"
    )
    print(path, flush=True)

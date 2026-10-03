# SPDX-License-Identifier: Apache-2.0
"""Experimental single-token BF16 FFN: four W1 and four/eight W2 workers.

Isolated artifact root is mandatory. Not a production compiler.
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

VALUE_CORES = int(os.environ.get("RWKV_XDNA_FFN_VALUE_CORES", "8"))
if VALUE_CORES not in (4, 8):
    raise ValueError("This experiment supports 4 or 8 W2 workers")
VALUE_ROWS = 2048 // VALUE_CORES
VALUE_GROUP = VALUE_CORES // 2
TRACE_ALL_MATRIX = os.environ.get("RWKV_XDNA_TRACE_ALL_MATRIX", "0") == "1"

# Offline diagnostic build only. Tracing adds a trailing BO to the ABI.
TRACE_ACTIVATION = os.environ.get("RWKV_XDNA_TRACE_ACTIVATION", "0") == "1"
TRACE_BYTES = int(os.environ.get("RWKV_XDNA_TRACE_BYTES", "0"))
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
        "channel_mix_spatial_bf16.cc",
        [typ(2048), wt(tile), typ(2048), np.int32, np.int32],
        optimization="-O3",
        stack_size=8192,
    )
    value = key.object_file.bind(
        "rwkv7_ffn_value_tile",
        [typ(8192), wt(tile), typ(VALUE_ROWS), np.int32, np.int32],
    )
    zero = key.object_file.bind("rwkv7_ffn_zero2048", [typ(2048)])
    vz = key.object_file.bind(
        "rwkv7_ffn_zero256" if VALUE_CORES == 8 else "rwkv7_channel_zero512",
        [typ(VALUE_ROWS)],
    )
    relu = key.object_file.bind("rwkv7_ffn_activate2048", [typ(2048), typ(2048)])
    add = key.object_file.bind("rwkv7_ffn_residual1024", [typ(2048), typ(2048)])
    norm_in = ObjectFifo(typ(8192), name="norm_input", depth=1)
    norm_join = norm_in.prod().join(
        [0, 2048, 6144], obj_types=[typ(2048), typ(4096), typ(2048)]
    )
    pair_norm = ObjectFifo(typ(4096), name="norm_pair", depth=1)
    packed_parameters = ObjectFifo(typ(6144), name="parameters", depth=1)
    parameter_parts = packed_parameters.cons().split(
        [0, 4096], obj_types=[typ(4096), typ(2048)]
    )
    copy4096 = key.object_file.bind("rwkv7_ffn_copy4096", [typ(4096), typ(4096)])
    copy1024 = key.object_file.bind("rwkv7_ffn_copy1024", [typ(1024), typ(1024)])
    norm = external("rwkv7_norm_pair", "norm_mix_fp32.cc", [typ(8192), typ(4096)])
    mix = external(
        "rwkv7_mix_pair", "mix_pair_fp32.cc", [typ(4096), typ(2048), typ(2048)]
    )
    fx = ObjectFifo(typ(2048), name="input", depth=1)
    w1 = [ObjectFifo(wt(tile), name=f"keyw{i}", depth=2) for i in range(4)]
    raw = ObjectFifo(typ(8192), name="raw", depth=1)
    act = ObjectFifo(typ(8192), name="active", depth=1)
    kr = raw.prod().join([i * 2048 for i in range(4)], obj_types=[typ(2048)] * 4)
    ka = act.prod().join([i * 2048 for i in range(4)], obj_types=[typ(2048)] * 4)
    w2 = [ObjectFifo(wt(tile), name=f"valuew{i}", depth=2) for i in range(VALUE_CORES)]
    pairs = [ObjectFifo(typ(2048), name=f"residual_pair{i}", depth=1) for i in range(2)]
    joins = [
        f.prod().join(
            [i * VALUE_ROWS for i in range(VALUE_GROUP)] + [1024],
            obj_types=[typ(VALUE_ROWS)] * VALUE_GROUP + [typ(1024)],
        )
        for f in pairs
    ]
    vo = [f for group in joins for f in group[:VALUE_GROUP]]
    residual = ObjectFifo(typ(2048), name="residual", depth=1)
    residual_parts = residual.cons().split([0, 1024], obj_types=[typ(1024)] * 2)
    final = ObjectFifo(typ(4096), name="result", depth=1)
    finals = final.prod().join([0, 2048], obj_types=[typ(2048)] * 2)

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
        for row in range_(VALUE_ROWS // 16):
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
            stack_size=12288,
        ),
        Worker(
            mixes,
            [pair_norm.cons(), parameter_parts[1].cons(), fx.prod(), mix],
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
                act.cons(),
                w2[i].cons(),
                vo[i].prod(),
                value,
                vz,
            ],
            stack_size=matrix_stack,
        )
        for i in range(VALUE_CORES)
    ]
    workers += [
        Worker(finish, [pairs[i].cons(), finals[i].prod(), add], stack_size=12288)
        for i in range(2)
    ]

    workers += [
        Worker(normalize, [parameter_parts[0].cons(), norm_join[1].prod(), copy4096])
    ]
    workers += [
        Worker(
            normalize,
            [residual_parts[i].cons(), joins[i][VALUE_GROUP].prod(), copy1024],
        )
        for i in range(2)
    ]

    def seq(
        x,
        parameters,
        w,
        diag,
        out,
        hx,
        hw,
        hold,
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
        hw.fill(parameters)
        hold.fill(diag, tap=TAP((22528,), 0, [1, 1, 1, 2048], [0, 0, 0, 1]))
        hn.drain(diag, tap=TAP((22528,), 0, [1, 1, 1, 4096], [0, 0, 0, 1]), wait=True)
        hm.drain(
            diag, tap=TAP((22528,), 4096, [1, 1, 1, 2048], [0, 0, 0, 1]), wait=True
        )
        for i in range(4):
            hw1[i].fill(
                w, tap=TAP((weight_count,), i * stripe, [1, 1, 1, stripe], [0, 0, 0, 1])
            )
        for i in range(VALUE_CORES):
            hw2[i].fill(
                w,
                tap=TAP(
                    (weight_count,),
                    weight_count // 2 + i * (weight_count // 2 // VALUE_CORES),
                    [1, 1, 1, weight_count // 2 // VALUE_CORES],
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
        # Local outputs are [projection1024, residual1024] per group.
        ho.drain(
            out, tap=TAP((4096,), 0, [1, 2, 2, 1024], [0, 1024, 2048, 1]), wait=True
        )

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
                norm_join[0].prod(),
                packed_parameters.prod(),
                norm_join[2].prod(),
                pair_norm.cons(),
                fx.cons(),
                [f.prod() for f in w1],
                [f.prod() for f in w2],
                raw.cons(),
                act.cons(),
                residual.prod(),
                final.cons(),
            ],
        ),
        workers=workers,
    )
    if TRACE_BYTES:
        # One key core and one value core: vector issue, memory/stream/lock
        # stalls and DMA port activity, using the compiler's default events.
        indices = [
            int(i) for i in os.environ.get("RWKV_XDNA_TRACE_WORKERS", "2,6").split(",")
        ]
        if any(i < 2 or i >= 6 + VALUE_CORES for i in indices) or len(
            set(indices)
        ) != len(indices):
            raise ValueError("Trace worker indices must select distinct matrix workers")
        program.enable_trace(
            trace_size=TRACE_BYTES,
            egress_shim_col=int(os.environ.get("RWKV_XDNA_TRACE_EGRESS_COL", "3")),
            workers=(
                workers[2 : 6 + VALUE_CORES]
                if TRACE_ALL_MATRIX
                else [workers[2], workers[6]]
            ),
        )
    return program.resolve_program()


@iron.jit
def design(x: In, parameters: In, weights: In, diagnostic: InOut, result: Out):
    return channel_mix_program()


if __name__ == "__main__":
    if not os.environ.get("RWKV_XDNA_KERNEL_DIR"):
        raise ValueError("Set RWKV_XDNA_KERNEL_DIR to an isolated experimental root")
    if KERNEL_ROOT.resolve() == (ROOT / "build/kernels/rwkv7-bf16").resolve():
        raise ValueError(
            "Experimental compiler must not overwrite the default production root"
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
                value_cores=VALUE_CORES,
                experimental=f"spatial_w2_{VALUE_CORES}_local_gather",
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

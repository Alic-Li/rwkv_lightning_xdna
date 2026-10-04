# SPDX-License-Identifier: Apache-2.0
"""R/K/V and W/A/G/(V) in one 14-core BF16 program, 16 shim input streams."""

import argparse
import json
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_common import C, KERNEL_ROOT, typ, external


def projections(count, batch=1):
    arena_stride = (30 * C) if batch == 2 and count == 4 else (27 * C)
    value_stride = arena_stride if batch == 2 and count == 4 else (3 * C)
    value_size = (batch - 1) * value_stride + (3 * C)
    source = (
        "prefill_attention_bf16.cc" if batch == 2 else "attention_projections_bf16.cc"
    )
    wt = lambda n: np.ndarray[(n,), np.dtype[bfloat16]]
    rkv_weight_type = wt(4096)
    n_rkv = 3 * C * C
    n_rank = count * 256 * C
    n_weights = n_rkv + 2 * n_rank

    @iron.jit
    def design(mixed: In, weights: In, arena: Out, value_aux: Out, rank_aux: Out):
        fn = external(
            (
                "rwkv7_prefill_attention_rank"
                if batch == 2
                else "rwkv7_attention_rank_tile"
            ),
            source,
            [typ(C * batch), wt(4096), typ(128 * batch), np.int32, np.int32],
            optimization="-O3",
        )
        z = fn.object_file.bind(
            (
                "rwkv7_prefill_attention_zero256"
                if batch == 2
                else "rwkv7_attention_zero128"
            ),
            [typ(128 * batch)],
        )
        act = fn.object_file.bind(
            (
                "rwkv7_prefill_attention_activate"
                if batch == 2
                else "rwkv7_attention_activate128"
            ),
            [typ(128 * batch), typ(128 * batch), np.int32],
        )
        mul = fn.object_file.bind(
            "rwkv7_prefill_attention_second" if batch == 2 else "rwkv7_gemv_tile",
            [typ(256 * batch), wt(4096), typ(16 * batch)],
        )
        rz = fn.object_file.bind(
            "rwkv7_prefill_attention_zero32" if batch == 2 else "rwkv7_zero",
            [typ(16 * batch)],
        )
        rkv = fn.object_file.bind(
            "rwkv7_prefill_attention_rkv" if batch == 2 else "rwkv7_attention_rkv_tile",
            [typ(C * batch), rkv_weight_type, typ(16 * batch), np.int32],
        )
        rx = ObjectFifo(typ(C * batch), name="rkv_x", depth=1)
        kx = ObjectFifo(typ(C * batch), name="rank_x", depth=1)
        rw = [ObjectFifo(rkv_weight_type, name=f"rkv_w{i}", depth=2) for i in range(8)]
        ry = [ObjectFifo(typ(16 * batch), name=f"rkv_y{i}", depth=2) for i in range(8)]
        w1 = [ObjectFifo(wt(4096), name=f"first_w{i}", depth=1) for i in range(2)]
        w2 = [ObjectFifo(wt(4096), name=f"second_w{i}", depth=1) for i in range(4)]
        raw = ObjectFifo(typ(256 * batch), name="raw", depth=1)
        activated = ObjectFifo(typ(256 * batch), name="active", depth=1)
        hr = raw.prod().join([0, 128 * batch], obj_types=[typ(128 * batch)] * 2)
        ha = activated.prod().join([0, 128 * batch], obj_types=[typ(128 * batch)] * 2)
        ky = [ObjectFifo(typ(16 * batch), name=f"rank_y{i}", depth=1) for i in range(4)]

        def rkv_core(x, w, y, f, z):
            for _ in range_(3):
                xv = x.acquire(1)
                for _ in range_(C // 128):
                    out = y.acquire(1)
                    z(out)
                    for col in range_(C // 256):
                        wv = w.acquire(1)
                        f(xv, wv, out, col)
                        w.release(1)
                    y.release(1)
                x.release(1)

        def first(x, w, r, a, f, z, activation):
            for p in range_(count):
                xv, rv, av = (x.acquire(1), r.acquire(1), a.acquire(1))
                z(rv)
                for row in range_(8):
                    for col in range_(C // 256):
                        wv = w.acquire(1)
                        f(xv, wv, rv, row, col)
                        w.release(1)
                activation(rv, av, p)
                x.release(1)
                r.release(1)
                a.release(1)

        def second(x, w, y, f, z):
            for _ in range_(count):
                xv = x.acquire(1)
                for _ in range_(C // 64):
                    wv, yv = (w.acquire(1), y.acquire(1))
                    z(yv)
                    f(xv, wv, yv)
                    w.release(1)
                    y.release(1)
                x.release(1)

        workers = [
            Worker(
                rkv_core,
                [rx.cons(), rw[i].cons(), ry[i].prod(), rkv, rz] + [],
                stack_size=12288,
                **{},
            )
            for i in range(8)
        ]
        workers += [
            Worker(
                first,
                [kx.cons(), w1[i].cons(), hr[i].prod(), ha[i].prod(), fn, z, act],
                stack_size=12288,
            )
            for i in range(2)
        ]
        workers += [
            Worker(
                second,
                [activated.cons(), w2[i].cons(), ky[i].prod(), mul, rz],
                stack_size=12288,
            )
            for i in range(4)
        ]

        def seq(
            x,
            w,
            arena,
            value_aux,
            rank_aux,
            hxr,
            hxk,
            hwr,
            hyr,
            hw1,
            hw2,
            hraw,
            hact,
            hyk,
        ):
            for slot in [0, 2, 3]:
                hxr.fill(
                    x,
                    tap=TAP(
                        ((6 * C) * batch,),
                        slot * C,
                        [batch, 1, 1, C],
                        [(6 * C), 0, 0, 1],
                    ),
                )
            for slot in [1, 4, 5, 3][:count]:
                hxk.fill(
                    x,
                    tap=TAP(
                        ((6 * C) * batch,),
                        slot * C,
                        [batch, 1, 1, C],
                        [(6 * C), 0, 0, 1],
                    ),
                )
            for i in range(8):
                hwr[i].fill(
                    w,
                    tap=TAP(
                        (n_weights,),
                        i * 3 * (C // 8) * C,
                        [1, 1, 1, 3 * (C // 8) * C],
                        [0, 0, 0, 1],
                    ),
                )
                destinations = [
                    (arena, arena_stride, 13 * C),
                    (arena, arena_stride, 0),
                    (
                        (arena, arena_stride, 16 * C)
                        if count == 3
                        else (value_aux, value_stride, 0)
                    ),
                ]
                for target, size, offset in destinations:
                    hyr[i].drain(
                        target,
                        tap=TAP(
                            (arena_stride * batch if target is arena else value_size,),
                            offset + i * (C // 8),
                            [C // 128, batch, 1, 16],
                            [16, size, 0, 1],
                        ),
                        wait=True,
                    )
            for i in range(2):
                hw1[i].fill(
                    w,
                    tap=TAP(
                        (n_weights,),
                        n_rkv + i * count * 128 * C,
                        [1, 1, 1, count * 128 * C],
                        [0, 0, 0, 1],
                    ),
                )
            hraw.drain(
                rank_aux,
                tap=TAP(
                    (count * 512 * batch,),
                    0,
                    [count, 2, batch, 128],
                    [256, 128, count * 512, 1],
                ),
                wait=True,
            )
            hact.drain(
                rank_aux,
                tap=TAP(
                    (count * 512 * batch,),
                    count * 256,
                    [count, 2, batch, 128],
                    [256, 128, count * 512, 1],
                ),
                wait=True,
            )
            for i in range(4):
                hw2[i].fill(
                    w,
                    tap=TAP(
                        (n_weights,),
                        n_rkv + n_rank + i * count * (C // 4) * 256,
                        [1, 1, 1, count * (C // 4) * 256],
                        [0, 0, 0, 1],
                    ),
                )
                destinations = [
                    (arena, arena_stride, 2 * C),
                    (arena, arena_stride, C),
                    (arena, arena_stride, 12 * C),
                    (value_aux, value_stride, C),
                ]
                for target, size, offset in destinations[:count]:
                    hyk[i].drain(
                        target,
                        tap=TAP(
                            (arena_stride * batch if target is arena else value_size,),
                            offset + i * (C // 4),
                            [C // 64, batch, 1, 16],
                            [16, size, 0, 1],
                        ),
                        wait=True,
                    )

        return Program(
            iron.get_current_device(),
            Runtime(
                seq,
                [
                    *[
                        typ((6 * C) * batch),
                        wt(n_weights + 0),
                        typ(arena_stride * batch),
                        typ(value_size),
                        typ(count * 512 * batch),
                    ],
                    rx.prod(),
                    kx.prod(),
                    [f.prod() for f in rw],
                    [f.cons() for f in ry],
                    [f.prod() for f in w1],
                    [f.prod() for f in w2],
                    raw.cons(),
                    activated.cons(),
                    [f.cons() for f in ky],
                ],
            ),
            workers=workers,
        ).resolve_program()

    return design


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    for count in (3, 4):
        batch = args.batch
        path = KERNEL_ROOT / (
            f"bf16-attention-projections-{count}" + ("-b2" if batch == 2 else "")
        )
        path.mkdir(parents=True, exist_ok=True)
        projections(count, batch).compile(
            path / "design.xclbin", path / "instructions.bin"
        )
        config = dict(
            schema_version=1,
            dtype="bfloat16",
            channels=C,
            branches=count,
            rank=256,
            cores=14,
            exact_fp32=False,
        )
        if batch == 2:
            config.update(
                batch=2,
                arena_stride=(30 * C) if count == 4 else (27 * C),
                value_stride=(30 * C) if count == 4 else (3 * C),
            )
        (path / "config.json").write_text(json.dumps(config) + "\n")
        print(path, flush=True)

# SPDX-License-Identifier: Apache-2.0
"""R/K/V and W/A/G/(V) in one 14-core BF16 program, 16 shim input streams."""

import json
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_common import KERNEL_ROOT, typ, external


def projections(count):
    wt = lambda n: np.ndarray[(n,), np.dtype[bfloat16]]
    n_rkv = 3 * 2048 * 2048
    n_rank = count * 256 * 2048
    n_weights = n_rkv + 2 * n_rank

    @iron.jit
    def design(mixed: In, weights: In, arena: Out, value_aux: Out, rank_aux: Out):
        fn = external(
            "rwkv7_attention_rank_tile",
            "attention_projections_bf16.cc",
            [typ(2048), wt(4096), typ(128), np.int32, np.int32],
            optimization="-O3",
        )
        z = fn.object_file.bind("rwkv7_attention_zero128", [typ(128)])
        act = fn.object_file.bind(
            "rwkv7_attention_activate128", [typ(128), typ(128), np.int32]
        )
        mul = fn.object_file.bind("rwkv7_gemv_tile", [typ(256), wt(4096), typ(16)])
        rz = fn.object_file.bind("rwkv7_zero", [typ(16)])
        rkv = fn.object_file.bind(
            "rwkv7_attention_rkv_tile", [typ(2048), wt(4096), typ(16), np.int32]
        )
        rx = ObjectFifo(typ(2048), name="rkv_x", depth=1)
        kx = ObjectFifo(typ(2048), name="rank_x", depth=1)
        rw = [ObjectFifo(wt(4096), name=f"rkv_w{i}", depth=2) for i in range(8)]
        ry = [ObjectFifo(typ(16), name=f"rkv_y{i}", depth=2) for i in range(8)]
        w1 = [ObjectFifo(wt(4096), name=f"first_w{i}", depth=1) for i in range(2)]
        w2 = [ObjectFifo(wt(4096), name=f"second_w{i}", depth=1) for i in range(4)]
        raw = ObjectFifo(typ(256), name="raw", depth=1)
        activated = ObjectFifo(typ(256), name="active", depth=1)
        hr = raw.prod().join([0, 128], obj_types=[typ(128)] * 2)
        ha = activated.prod().join([0, 128], obj_types=[typ(128)] * 2)
        ky = [ObjectFifo(typ(16), name=f"rank_y{i}", depth=1) for i in range(4)]

        def rkv_core(x, w, y, f, z):
            for _ in range_(3):
                xv = x.acquire(1)
                for _ in range_(16):
                    out = y.acquire(1)
                    z(out)
                    for col in range_(8):
                        wv = w.acquire(1)
                        f(xv, wv, out, col)
                        w.release(1)
                    y.release(1)
                x.release(1)

        def first(x, w, r, a, f, z, activation):
            for p in range_(count):
                xv, rv, av = x.acquire(1), r.acquire(1), a.acquire(1)
                z(rv)
                for row in range_(8):
                    for col in range_(8):
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
                for _ in range_(32):
                    wv, yv = w.acquire(1), y.acquire(1)
                    z(yv)
                    f(xv, wv, yv)
                    w.release(1)
                    y.release(1)
                x.release(1)

        workers = [
            Worker(
                rkv_core,
                [rx.cons(), rw[i].cons(), ry[i].prod(), rkv, rz],
                stack_size=12288,
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
                    x, tap=TAP((12288,), slot * 2048, [1, 1, 1, 2048], [0, 0, 0, 1])
                )
            for slot in [1, 4, 5, 3][:count]:
                hxk.fill(
                    x, tap=TAP((12288,), slot * 2048, [1, 1, 1, 2048], [0, 0, 0, 1])
                )
            for i in range(8):
                hwr[i].fill(
                    w,
                    tap=TAP(
                        (n_weights,),
                        i * 3 * 256 * 2048,
                        [1, 1, 1, 3 * 256 * 2048],
                        [0, 0, 0, 1],
                    ),
                )
                destinations = [
                    (arena, 55296, 13 * 2048),
                    (arena, 55296, 0),
                    (arena, 55296, 16 * 2048) if count == 3 else (value_aux, 6144, 0),
                ]
                for target, size, offset in destinations:
                    hyr[i].drain(
                        target,
                        tap=TAP(
                            (size,), offset + i * 256, [1, 1, 1, 256], [0, 0, 0, 1]
                        ),
                        wait=True,
                    )
            for i in range(2):
                hw1[i].fill(
                    w,
                    tap=TAP(
                        (n_weights,),
                        n_rkv + i * count * 128 * 2048,
                        [1, 1, 1, count * 128 * 2048],
                        [0, 0, 0, 1],
                    ),
                )
            hraw.drain(
                rank_aux,
                tap=TAP((count * 512,), 0, [1, 1, 1, count * 256], [0, 0, 0, 1]),
                wait=True,
            )
            hact.drain(
                rank_aux,
                tap=TAP(
                    (count * 512,), count * 256, [1, 1, 1, count * 256], [0, 0, 0, 1]
                ),
                wait=True,
            )
            for i in range(4):
                hw2[i].fill(
                    w,
                    tap=TAP(
                        (n_weights,),
                        n_rkv + n_rank + i * count * 512 * 256,
                        [1, 1, 1, count * 512 * 256],
                        [0, 0, 0, 1],
                    ),
                )
                destinations = [
                    (arena, 55296, 4096),
                    (arena, 55296, 2048),
                    (arena, 55296, 12 * 2048),
                    (value_aux, 6144, 2048),
                ]
                for target, size, offset in destinations[:count]:
                    hyk[i].drain(
                        target,
                        tap=TAP(
                            (size,), offset + i * 512, [1, 1, 1, 512], [0, 0, 0, 1]
                        ),
                        wait=True,
                    )

        return Program(
            iron.get_current_device(),
            Runtime(
                seq,
                [
                    typ(12288),
                    wt(n_weights),
                    typ(55296),
                    typ(6144),
                    typ(count * 512),
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
    for count in [3, 4]:
        path = KERNEL_ROOT / f"bf16-attention-projections-{count}"
        path.mkdir(parents=True, exist_ok=True)
        projections(count).compile(path / "design.xclbin", path / "instructions.bin")
        (path / "config.json").write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    dtype="bfloat16",
                    channels=2048,
                    branches=count,
                    rank=256,
                    cores=14,
                    exact_fp32=False,
                )
            )
            + "\n"
        )
        print(path, flush=True)

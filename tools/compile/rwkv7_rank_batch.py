# SPDX-License-Identifier: Apache-2.0
"""Batch independent W/A/G/(V) low-rank branches into one resident program."""

import json
import os
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_full import KERNEL_ROOT, typ, external


def batch(count):
    wt = lambda n: np.ndarray[(n,), np.dtype[bfloat16]]

    @iron.jit
    def design(mixed: In, weights: In, auxiliary: Out, wa: Out, yg: Out, yv: Out):
        fn = external(
            "rwkv7_rank_tile64",
            "rank_bf16.cc",
            [typ(256), wt(4096), typ(64), np.int32],
            optimization="-O3",
        )
        zero = fn.object_file.bind("rwkv7_rank_zero64", [typ(64)])
        act = fn.object_file.bind(
            "rwkv7_rank_batch_activate64", [typ(64), typ(64), np.int32]
        )
        mul = fn.object_file.bind("rwkv7_gemv_tile", [typ(256), wt(4096), typ(16)])
        init = fn.object_file.bind("rwkv7_zero", [typ(16)])
        xs = [ObjectFifo(typ(256), name=f"x{i}", depth=1) for i in range(4)]
        ws1 = [ObjectFifo(wt(4096), name=f"w1_{i}", depth=1) for i in range(4)]
        rawfifo = ObjectFifo(typ(256), name="raw", depth=1)
        active = ObjectFifo(typ(256), name="active", depth=1)
        raws = rawfifo.prod().join([0, 64, 128, 192], obj_types=[typ(64)] * 4)
        acts = active.prod().join([0, 64, 128, 192], obj_types=[typ(64)] * 4)
        ws2 = [ObjectFifo(wt(4096), name=f"w2_{i}", depth=1) for i in range(8)]
        ys = [ObjectFifo(typ(16), name=f"y{i}", depth=1) for i in range(8)]

        def first(x, w, h, a, f, z, activation):
            for projection in range_(count):
                out, ov = h.acquire(1), a.acquire(1)
                z(out)
                for row in range_(4):
                    for _ in range_(8):
                        xv, wv = x.acquire(1), w.acquire(1)
                        f(xv, wv, out, row)
                        x.release(1)
                        w.release(1)
                activation(out, ov, projection)
                h.release(1)
                a.release(1)

        def second(x, w, y, f, z):
            for _ in range_(count):
                xv = x.acquire(1)
                for _ in range_(16):
                    wv, out = w.acquire(1), y.acquire(1)
                    z(out)
                    f(xv, wv, out)
                    w.release(1)
                    y.release(1)
                x.release(1)

        workers = [
            Worker(
                first,
                [
                    xs[i].cons(),
                    ws1[i].cons(),
                    raws[i].prod(),
                    acts[i].prod(),
                    fn,
                    zero,
                    act,
                ],
                stack_size=12288,
            )
            for i in range(4)
        ]
        workers += [
            Worker(
                second,
                [active.cons(), ws2[i].cons(), ys[i].prod(), mul, init],
                stack_size=12288,
            )
            for i in range(8)
        ]

        def seq(x, weights, auxiliary, wa, yg, yv, hx, hw1, hr, ha, hw2, hy):
            for i in range(4):
                for slot in [1, 4, 5, 3][:count]:
                    hx[i].fill(
                        x, tap=TAP((12288,), slot * 2048, [4, 1, 1, 2048], [0, 0, 0, 1])
                    )
                hw1[i].fill(
                    weights,
                    tap=TAP(
                        (count * 2 * 256 * 2048,),
                        i * count * 64 * 2048,
                        [1, 1, 1, count * 64 * 2048],
                        [0, 0, 0, 1],
                    ),
                )
            hr.drain(
                auxiliary,
                tap=TAP((count * 512,), 0, [1, 1, 1, count * 256], [0, 0, 0, 1]),
                wait=True,
            )
            ha.drain(
                auxiliary,
                tap=TAP(
                    (count * 512,), count * 256, [1, 1, 1, count * 256], [0, 0, 0, 1]
                ),
                wait=True,
            )
            for i in range(8):
                hw2[i].fill(
                    weights,
                    tap=TAP(
                        (count * 2 * 2048 * 256,),
                        count * 256 * 2048 + i * count * 256 * 256,
                        [1, 1, 1, count * 256 * 256],
                        [0, 0, 0, 1],
                    ),
                )
                for out, size, offset in [
                    (wa, 16384, 4096),
                    (wa, 16384, 2048),
                    (yg, 2048, 0),
                    (yv, 2048, 0),
                ][:count]:
                    hy[i].drain(
                        out,
                        tap=TAP(
                            (size,), offset + i * 256, [1, 1, 1, 256], [0, 0, 0, 1]
                        ),
                        wait=True,
                    )

        return Program(
            iron.get_current_device(),
            Runtime(
                seq,
                [
                    typ(12288),
                    wt(count * 2 * 256 * 2048),
                    typ(count * 512),
                    typ(16384),
                    typ(2048),
                    typ(2048),
                    [f.prod() for f in xs],
                    [f.prod() for f in ws1],
                    rawfifo.cons(),
                    active.cons(),
                    [f.prod() for f in ws2],
                    [f.cons() for f in ys],
                ],
            ),
            workers=workers,
        ).resolve_program()

    return design


if __name__ == "__main__":
    for count in [3, 4]:
        path = KERNEL_ROOT / f"bf16-rank-batch-{count}"
        path.mkdir(parents=True, exist_ok=True)
        batch(count).compile(path / "design.xclbin", path / "instructions.bin")
        (path / "config.json").write_text(
            json.dumps(
                dict(
                    schema_version=2,
                    dtype="bfloat16",
                    channels=2048,
                    rank=256,
                    branches=count,
                    exact_fp32=os.environ.get("RWKV_XDNA_EXACT", "1") == "1",
                )
            )
            + "\n"
        )
        print(path, flush=True)

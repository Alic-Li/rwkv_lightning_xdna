# SPDX-License-Identifier: Apache-2.0
"""Three BF16 R/K/V projections in one program, reusing each DMA endpoint."""

import json
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_full import KERNEL_ROOT, typ, external


@iron.jit
def design(mixed: In, weights: In, r: Out, k: Out, v: Out):
    wt = lambda n: np.ndarray[(n,), np.dtype[bfloat16]]
    fn = external(
        "rwkv7_gemv_tile",
        "gemv_bf16.cc",
        [typ(256), wt(4096), typ(16)],
        optimization="-O3",
    )
    zero = fn.object_file.bind("rwkv7_zero", [typ(16)])
    xs = [ObjectFifo(typ(256), name=f"x{i}", depth=2) for i in range(8)]
    ws = [ObjectFifo(wt(4096), name=f"w{i}", depth=2) for i in range(8)]
    ys = [ObjectFifo(typ(16), name=f"y{i}", depth=2) for i in range(8)]

    def core(x, w, y, f, z):
        for _ in range_(3 * 16):
            out = y.acquire(1)
            z(out)
            for _ in range_(8):
                xv, wv = x.acquire(1), w.acquire(1)
                f(xv, wv, out)
                x.release(1)
                w.release(1)
            y.release(1)

    workers = [
        Worker(
            core, [xs[i].cons(), ws[i].cons(), ys[i].prod(), fn, zero], stack_size=12288
        )
        for i in range(8)
    ]

    def seq(x, w, r, k, v, hx, hw, hy):
        for i in range(8):
            # Mix ABI is [xr,xw,xk,xv,xa,xg]. These three source offsets
            # deliberately use the same endpoint, in consumer order.
            for slot in [0, 2, 3]:
                hx[i].fill(
                    x, tap=TAP((12288,), slot * 2048, [16, 1, 1, 2048], [0, 0, 0, 1])
                )
            hw[i].fill(
                w,
                tap=TAP(
                    (3 * 2048 * 2048,),
                    i * 3 * 256 * 2048,
                    [1, 1, 1, 3 * 256 * 2048],
                    [0, 0, 0, 1],
                ),
            )
            for out in [r, k, v]:
                hy[i].drain(
                    out,
                    tap=TAP((2048,), i * 256, [1, 1, 1, 256], [0, 0, 0, 1]),
                    wait=True,
                )

    return Program(
        iron.get_current_device(),
        Runtime(
            seq,
            [
                typ(12288),
                wt(3 * 2048 * 2048),
                typ(2048),
                typ(2048),
                typ(2048),
                [f.prod() for f in xs],
                [f.prod() for f in ws],
                [f.cons() for f in ys],
            ],
        ),
        workers=workers,
    ).resolve_program()


if __name__ == "__main__":
    path = KERNEL_ROOT / "bf16-rkv"
    path.mkdir(parents=True, exist_ok=True)
    design.compile(path / "design.xclbin", path / "instructions.bin")
    (path / "config.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                dtype="bfloat16",
                channels=2048,
                cores=8,
                layout="worker_projection_row_tile_k_tile",
                input_slots=[0, 2, 3],
            )
        )
        + "\n"
    )
    print(path)

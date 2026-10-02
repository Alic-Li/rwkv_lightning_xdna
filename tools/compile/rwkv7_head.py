# SPDX-License-Identifier: Apache-2.0
"""Single-run BF16 vocabulary projection; keep the input vector on each core."""

import json
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_full import KERNEL_ROOT, typ, external


@iron.jit
def design(x: In, w: In, y: Out):
    wt = lambda n: np.ndarray[(n,), np.dtype[bfloat16]]
    fn = external(
        "rwkv7_head_tile",
        "head_bf16.cc",
        [typ(2048), wt(4096), typ(16), np.int32],
        optimization="-O3",
    )
    zero = fn.object_file.bind("rwkv7_zero", [typ(16)])
    xs = [ObjectFifo(typ(2048), name=f"x{i}", depth=1) for i in range(8)]
    ws = [ObjectFifo(wt(4096), name=f"w{i}", depth=2) for i in range(8)]
    ys = [ObjectFifo(typ(16), name=f"y{i}", depth=2) for i in range(8)]

    def core(x, w, y, f, z):
        xv = x.acquire(1)
        for _ in range_(512):
            out = y.acquire(1)
            z(out)
            for col in range_(8):
                wv = w.acquire(1)
                f(xv, wv, out, col)
                w.release(1)
            y.release(1)
        x.release(1)

    workers = [
        Worker(
            core, [xs[i].cons(), ws[i].cons(), ys[i].prod(), fn, zero], stack_size=12288
        )
        for i in range(8)
    ]

    def seq(x, w, y, hx, hw, hy):
        for i in range(8):
            hx[i].fill(x)
            hw[i].fill(
                w,
                tap=TAP((134217728,), i * 16777216, [1, 1, 1, 16777216], [0, 0, 0, 1]),
            )
            hy[i].drain(
                y, tap=TAP((65536,), i * 8192, [1, 1, 1, 8192], [0, 0, 0, 1]), wait=True
            )

    return Program(
        iron.get_current_device(),
        Runtime(
            seq,
            [
                typ(2048),
                wt(134217728),
                typ(65536),
                [f.prod() for f in xs],
                [f.prod() for f in ws],
                [f.cons() for f in ys],
            ],
        ),
        workers=workers,
    ).resolve_program()


if __name__ == "__main__":
    path = KERNEL_ROOT / "bf16-array-gemv-2048-65536"
    path.mkdir(parents=True, exist_ok=True)
    design.compile(path / "design.xclbin", path / "instructions.bin")
    (path / "config.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                dtype="bfloat16",
                k=2048,
                rows=65536,
                cores=8,
                input_resident=True,
            )
        )
        + "\n"
    )
    print(path, flush=True)

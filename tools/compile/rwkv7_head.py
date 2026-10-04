# SPDX-License-Identifier: Apache-2.0
"""Single-run BF16 vocabulary projection; keep the input vector on each core."""

import json
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_common import C, V, KERNEL_ROOT, typ, external


@iron.jit
def design(x: In, w: In, y: Out):
    wt = lambda n: np.ndarray[(n,), np.dtype[bfloat16]]
    fn = external(
        "rwkv7_head_tile",
        "head_bf16.cc",
        [typ(C), wt(4096), typ(16), np.int32],
        optimization="-O3",
    )
    zero = fn.object_file.bind("rwkv7_zero", [typ(16)])
    xs = [ObjectFifo(typ(C), name=f"x{i}", depth=1) for i in range(8)]
    ws = [ObjectFifo(wt(4096), name=f"w{i}", depth=2) for i in range(8)]
    ys = [ObjectFifo(typ(16), name=f"y{i}", depth=2) for i in range(8)]

    def core(x, w, y, f, z):
        xv = x.acquire(1)
        for _ in range_(V // 128):
            out = y.acquire(1)
            z(out)
            for col in range_(C // 256):
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
                tap=TAP(
                    ((C * V),), i * (C * V // 8), [1, 1, 1, (C * V // 8)], [0, 0, 0, 1]
                ),
            )
            hy[i].drain(
                y,
                tap=TAP((V,), i * (V // 8), [1, 1, 1, (V // 8)], [0, 0, 0, 1]),
                wait=True,
            )

    return Program(
        iron.get_current_device(),
        Runtime(
            seq,
            [
                typ(C),
                wt((C * V)),
                typ(V),
                [f.prod() for f in xs],
                [f.prod() for f in ws],
                [f.cons() for f in ys],
            ],
        ),
        workers=workers,
    ).resolve_program()


if __name__ == "__main__":
    path = KERNEL_ROOT / f"bf16-array-gemv-{C}-{V}"
    path.mkdir(parents=True, exist_ok=True)
    design.compile(path / "design.xclbin", path / "instructions.bin")
    (path / "config.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                dtype="bfloat16",
                k=C,
                rows=V,
                cores=8,
                input_resident=True,
            )
        )
        + "\n"
    )
    print(path, flush=True)

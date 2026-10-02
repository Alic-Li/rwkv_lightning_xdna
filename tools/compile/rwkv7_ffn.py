# SPDX-License-Identifier: Apache-2.0
"""FP32 array projections: independent output stripes on eight AIE workers."""

import argparse
import json
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_full import ROOT, typ, external as base_external


def external(*args):
    return base_external(*args, optimization="-O3")


def ffn(k=2048, rows=8192, cores=8):
    stripe = rows // cores

    @iron.jit
    def design(x: In, w: In, y: Out, activated: Out):
        fn = external("rwkv7_gemv_tile", "gemv_fp32.cc", [typ(256), typ(4096), typ(32)])
        zero = fn.object_file.bind("rwkv7_zero", [typ(32)])
        relu = fn.object_file.bind("rwkv7_relu_squared", [typ(32)])
        xs = [ObjectFifo(typ(256), name=f"x{i}", depth=2) for i in range(cores)]
        ws = [ObjectFifo(typ(4096), name=f"w{i}", depth=2) for i in range(cores)]
        ys = [ObjectFifo(typ(32), name=f"y{i}", depth=2) for i in range(cores)]
        splits = [f.cons().split([0, 16], obj_types=[typ(16)] * 2) for f in ys]

        def core(ix, iw, oy, mul, init, relu):
            for _ in range_(stripe // 16):
                out = oy.acquire(1)
                init(out)
                for _ in range_(k // 256):
                    a, b = ix.acquire(1), iw.acquire(1)
                    mul(a, b, out)
                    ix.release(1)
                    iw.release(1)
                relu(out)
                oy.release(1)

        workers = [
            Worker(
                core,
                [xs[i].cons(), ws[i].cons(), ys[i].prod(), fn, zero, relu],
                stack_size=12288,
            )
            for i in range(cores)
        ]

        def seq(x, w, y, activated, hx, hw, hy, ha):
            for i in range(cores):
                hx[i].fill(x, tap=TAP((k,), 0, [stripe // 16, 1, 1, k], [0, 0, 0, 1]))
                hw[i].fill(
                    w,
                    tap=TAP(
                        (rows * k,), i * stripe * k, [1, 1, 1, stripe * k], [0, 0, 0, 1]
                    ),
                )
                hy[i].drain(
                    y,
                    tap=TAP((rows,), i * stripe, [1, 1, 1, stripe], [0, 0, 0, 1]),
                    wait=True,
                )
                ha[i].drain(
                    activated,
                    tap=TAP((rows,), i * stripe, [1, 1, 1, stripe], [0, 0, 0, 1]),
                    wait=True,
                )

        return Program(
            iron.get_current_device(),
            Runtime(
                seq,
                [
                    typ(k),
                    typ(rows * k),
                    typ(rows),
                    typ(rows),
                    [f.prod() for f in xs],
                    [f.prod() for f in ws],
                    [f[0].cons() for f in splits],
                    [f[1].cons() for f in splits],
                ],
            ),
            workers=workers,
        ).resolve_program()

    return design


if __name__ == "__main__":
    path = ROOT / "build/kernels/rwkv7-full/fused-ffn-key"
    path.mkdir(parents=True, exist_ok=True)
    ffn().compile(path / "design.xclbin", path / "instructions.bin")
    (path / "config.json").write_text(
        json.dumps(dict(schema_version=1, dtype="float32", rows=8192, k=2048)) + "\n"
    )

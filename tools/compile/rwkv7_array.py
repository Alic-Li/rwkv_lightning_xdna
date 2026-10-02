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


def gemv(k, rows=2048, cores=8):
    stripe = rows // cores

    @iron.jit
    def design(x: In, w: In, y: Out):
        fn = external("rwkv7_gemv_tile", "gemv_fp32.cc", [typ(256), typ(4096), typ(16)])
        zero = fn.object_file.bind("rwkv7_zero", [typ(16)])
        xs = [ObjectFifo(typ(256), name=f"x{i}", depth=2) for i in range(cores)]
        ws = [ObjectFifo(typ(4096), name=f"w{i}", depth=2) for i in range(cores)]
        ys = [ObjectFifo(typ(16), name=f"y{i}", depth=2) for i in range(cores)]

        def core(ix, iw, oy, mul, init):
            for _ in range_(stripe // 16):
                out = oy.acquire(1)
                init(out)
                for _ in range_(k // 256):
                    a, b = ix.acquire(1), iw.acquire(1)
                    mul(a, b, out)
                    ix.release(1)
                    iw.release(1)
                oy.release(1)

        workers = [
            Worker(
                core,
                [xs[i].cons(), ws[i].cons(), ys[i].prod(), fn, zero],
                stack_size=12288,
            )
            for i in range(cores)
        ]

        def seq(x, w, y, hx, hw, hy):
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

        return Program(
            iron.get_current_device(),
            Runtime(
                seq,
                [
                    typ(k),
                    typ(rows * k),
                    typ(rows),
                    [f.prod() for f in xs],
                    [f.prod() for f in ws],
                    [f.cons() for f in ys],
                ],
            ),
            workers=workers,
        ).resolve_program()

    return design


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, nargs="+", default=[256, 2048, 8192])
    p.add_argument("--rows", type=int, default=2048)
    p.add_argument("--output", type=str, default=str(ROOT / "build/kernels/rwkv7-full"))
    args = p.parse_args()
    from pathlib import Path

    for k in args.k:
        path = Path(args.output) / (
            f"array-gemv-{k}" if args.rows == 2048 else f"array-gemv-{k}-{args.rows}"
        )
        path.mkdir(parents=True, exist_ok=True)
        gemv(k, args.rows).compile(path / "design.xclbin", path / "instructions.bin")
        (path / "config.json").write_text(
            json.dumps(
                dict(schema_version=1, dtype="float32", rows=args.rows, cores=8, k=k)
            )
            + "\n"
        )
        print(path, flush=True)

# SPDX-License-Identifier: Apache-2.0
"""32-worker GEMV: four cores per column, broadcast x and split weight tiles."""

import argparse
import json
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_full import ROOT, typ, external


def gemv(k, rows):
    stripe = rows // 32

    @iron.jit
    def design(x: In, w: In, y: Out):
        fn = external(
            "rwkv7_gemv_tile",
            "gemv_fp32.cc",
            [typ(256), typ(4096), typ(16)],
            optimization="-O3",
        )
        zero = fn.object_file.bind("rwkv7_zero", [typ(16)])
        xs = [ObjectFifo(typ(256), name=f"x{i}", depth=1) for i in range(8)]
        ws = [ObjectFifo(typ(16384), name=f"w{i}", depth=1) for i in range(8)]
        ys = [ObjectFifo(typ(64), name=f"y{i}", depth=1) for i in range(8)]
        split = [
            f.cons().split([0, 4096, 8192, 12288], obj_types=[typ(4096)] * 4)
            for f in ws
        ]
        joined = [f.prod().join([0, 16, 32, 48], obj_types=[typ(16)] * 4) for f in ys]

        def core(x, w, y, f, z):
            for _ in range_(stripe // 16):
                out = y.acquire(1)
                z(out)
                for _ in range_(k // 256):
                    a, b = x.acquire(1), w.acquire(1)
                    f(a, b, out)
                    x.release(1)
                    w.release(1)
                y.release(1)

        workers = [
            Worker(
                core,
                [xs[c].cons(), split[c][r].cons(), joined[c][r].prod(), fn, zero],
                stack_size=12288,
            )
            for c in range(8)
            for r in range(4)
        ]

        def seq(x, w, y, hx, hw, hy):
            for c in range(8):
                hx[c].fill(x, tap=TAP((k,), 0, [stripe // 16, 1, 1, k], [0, 0, 0, 1]))
                hw[c].fill(
                    w,
                    tap=TAP(
                        (rows * k,),
                        c * 4 * stripe * k,
                        [1, 1, 1, 4 * stripe * k],
                        [0, 0, 0, 1],
                    ),
                )
                hy[c].drain(
                    y,
                    tap=TAP(
                        (rows,),
                        c * 4 * stripe,
                        [1, stripe // 16, 4, 16],
                        [0, 16, stripe, 1],
                    ),
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
    a = p.parse_args()
    for k in a.k:
        path = (
            ROOT
            / "build/kernels/rwkv7-full"
            / (f"array32-gemv-{k}" + (f"-{a.rows}" if a.rows != 2048 else ""))
        )
        path.mkdir(parents=True, exist_ok=True)
        gemv(k, a.rows).compile(path / "design.xclbin", path / "instructions.bin")
        (path / "config.json").write_text(
            json.dumps(
                dict(
                    schema_version=2,
                    dtype="float32",
                    k=k,
                    rows=a.rows,
                    cores=32,
                    layout="column_block_lane",
                )
            )
            + "\n"
        )
        print(path, flush=True)

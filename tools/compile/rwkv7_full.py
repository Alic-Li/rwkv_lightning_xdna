# SPDX-License-Identifier: Apache-2.0
"""Offline compiler for all-NPU RWKV-7 operations and tiled FP32 projections."""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import aie.iron as iron
from aie.iron import ExternalFunction, In, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron.device import NPU2
from aie.utils import set_current_device
from rwkv7 import design as decode

ROOT = Path(__file__).resolve().parents[2]
KERNEL_ROOT = Path(
    os.environ.get("RWKV_XDNA_KERNEL_DIR", ROOT / "build/kernels/rwkv7-full")
)
set_current_device(NPU2())


def typ(n):
    return np.ndarray[(n,), np.dtype[np.float32]]


def external(name, source, types, optimization="-Oz"):
    return ExternalFunction(
        name,
        source_file=str(ROOT / "kernels/rwkv" / source),
        arg_types=types,
        compile_flags=[
            optimization,
            "-fno-fast-math",
            "-ffp-contract=off",
            "-D__AIE_API_FP32_EMULATION__=1",
        ]
        + (
            ["-DRWKV_EXACT_FP32=1"]
            if os.environ.get("RWKV_XDNA_EXACT", "1") == "1"
            else []
        ),
        stack_size_override=8192,
    )


@iron.jit
def ops(x: In, y: Out):
    it, ot = typ(8208), typ(2048)
    fn = external("rwkv7_ops_fp32", "ops_fp32.cc", [it, ot])
    fi, fo = ObjectFifo(it, name="in", depth=1), ObjectFifo(ot, name="out", depth=1)

    def core(i, o, k):
        a, b = i.acquire(1), o.acquire(1)
        k(a, b)
        i.release(1)
        o.release(1)

    worker = Worker(core, [fi.cons(), fo.prod(), fn], stack_size=12288)

    def seq(x, y, hi, ho):
        hi.fill(x)
        ho.drain(y, wait=True)

    return Program(
        iron.get_current_device(),
        Runtime(seq, [it, ot, fi.prod(), fo.cons()]),
        workers=[worker],
    ).resolve_program()


@iron.jit
def prefill(state: In, packed: In, out: Out):
    st, pt, ot = typ(4096), typ(400), typ(5120)
    fn = external("rwkv7_prefill_step", "wkv7_prefill_fp32.cc", [pt, ot, np.int32])
    init = fn.object_file.bind("rwkv7_prefill_init", [st, ot])
    fs, fp, fo = (
        ObjectFifo(st, name="s", depth=1),
        ObjectFifo(pt, name="p", depth=1),
        ObjectFifo(ot, name="o", depth=1),
    )

    def core(s, p, o, step, initialize):
        sv, ov = s.acquire(1), o.acquire(1)
        initialize(sv, ov)
        s.release(1)
        for t in range_(16):
            pv = p.acquire(1)
            step(pv, ov, t)
            p.release(1)
        o.release(1)

    worker = Worker(core, [fs.cons(), fp.cons(), fo.prod(), fn, init], stack_size=12288)

    def seq(s, p, o, hs, hp, ho):
        hs.fill(s)
        hp.fill(p)
        ho.drain(o, wait=True)

    return Program(
        iron.get_current_device(),
        Runtime(seq, [st, typ(16 * 400), ot, fs.prod(), fp.prod(), fo.cons()]),
        workers=[worker],
    ).resolve_program()


def gemv(k):
    @iron.jit
    def design(x: In, w: In, y: Out):
        xt, wt, yt = typ(256), typ(4096), typ(16)
        fn = external("rwkv7_gemv_tile", "gemv_fp32.cc", [xt, wt, yt])
        zero = fn.object_file.bind("rwkv7_zero", [yt])
        fx, fw, fy = (
            ObjectFifo(xt, name="x", depth=1),
            ObjectFifo(wt, name="w", depth=1),
            ObjectFifo(yt, name="y", depth=1),
        )

        def core(ix, iw, oy, mul, init):
            for _ in range_(16):
                out = oy.acquire(1)
                init(out)
                for _ in range_(k // 256):
                    a, b = ix.acquire(1), iw.acquire(1)
                    mul(a, b, out)
                    ix.release(1)
                    iw.release(1)
                oy.release(1)

        worker = Worker(
            core, [fx.cons(), fw.cons(), fy.prod(), fn, zero], stack_size=12288
        )

        def seq(x, w, y, hx, hw, hy):
            hx.fill(x, tap=TensorAccessPattern((k,), 0, [16, 1, 1, k], [0, 0, 0, 1]))
            hw.fill(w)
            hy.drain(y, wait=True)

        return Program(
            iron.get_current_device(),
            Runtime(
                seq, [typ(k), typ(256 * k), typ(256), fx.prod(), fw.prod(), fy.cons()]
            ),
            workers=[worker],
        ).resolve_program()

    return design


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=KERNEL_ROOT)
    p.add_argument("--k", type=int, nargs="+", default=[256, 2048, 8192])
    args = p.parse_args()
    for name, design in [("ops", ops), ("prefill", prefill), ("decode", decode)] + [
        (f"gemv-{k}", gemv(k)) for k in args.k
    ]:
        directory = args.output / name
        directory.mkdir(parents=True, exist_ok=True)
        design.compile(directory / "design.xclbin", directory / "instructions.bin")
        if name == "decode":
            (directory / "config.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "head_size": 64,
                        "dtype": "float32",
                        "layout": "key_value",
                    },
                    indent=2,
                )
                + "\n"
            )
        print(name, flush=True)
    (args.output / "config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dtype": "float32",
                "vector_size": 2048,
                "gemv_rows": 256,
                "gemv_k": args.k,
            },
            indent=2,
        )
        + "\n"
    )

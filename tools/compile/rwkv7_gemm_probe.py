# SPDX-License-Identifier: Apache-2.0
"""Isolated native BF16 GEMM probe, 64x64x64, upstream 4x8x8 MMUL."""

import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Worker, Runtime, Program
from rwkv7_full import KERNEL_ROOT, typ, external


@iron.jit
def design(a: In, b: In, c: Out):
    bt = np.ndarray[(4096,), np.dtype[bfloat16]]
    ct = typ(4096)
    f = external(
        "rwkv7_gemm_probe", "gemm_probe_bf16.cc", [bt, bt, ct], optimization="-O3"
    )
    fa, fb, fc = [
        ObjectFifo(t, name=n, depth=1) for t, n in [(bt, "a"), (bt, "b"), (ct, "c")]
    ]

    def core(a, b, c, f):
        av, bv, cv = a.acquire(1), b.acquire(1), c.acquire(1)
        f(av, bv, cv)
        a.release(1)
        b.release(1)
        c.release(1)

    w = Worker(core, [fa.cons(), fb.cons(), fc.prod(), f], stack_size=12288)

    def seq(a, b, c, ha, hb, hc):
        ha.fill(a)
        hb.fill(b)
        hc.drain(c, wait=True)

    return Program(
        iron.get_current_device(),
        Runtime(seq, [bt, bt, ct, fa.prod(), fb.prod(), fc.cons()]),
        workers=[w],
    ).resolve_program()


if __name__ == "__main__":
    path = KERNEL_ROOT / "bf16-gemm-probe"
    path.mkdir(parents=True, exist_ok=True)
    design.compile(path / "design.xclbin", path / "instructions.bin")

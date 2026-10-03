# SPDX-License-Identifier: Apache-2.0
"""Two-token BF16 output projection with weight reuse and residual addition."""
import json
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_common import KERNEL_ROOT, typ, external

@iron.jit
def design(x: In, w: In, residual0: In, residual1: In, result: Out, *, stride: CompileTime[int]):
    if stride not in (2048, 55296, 61440):
        raise ValueError("Unsupported input stride")
    wt = lambda n: np.ndarray[(n,), np.dtype[bfloat16]]
    fn = external("rwkv7_prefill_output_tile", "prefill_output_bf16.cc",
                  [typ(4096), wt(4096), typ(512), np.int32, np.int32], optimization="-O3")
    zero = fn.object_file.bind("rwkv7_prefill_output_zero", [typ(512)])
    add = fn.object_file.bind("rwkv7_prefill_output_residual", [typ(512), typ(2048), typ(2048), typ(1024), np.int32])
    xs = ObjectFifo(typ(4096), name="input", depth=1)
    ws = [ObjectFifo(wt(4096), name=f"w{i}", depth=2) for i in range(8)]
    residuals = [ObjectFifo(typ(2048), name=f"residual{i}", depth=1) for i in range(2)]
    ps = [ObjectFifo(typ(512), name=f"p{i}", depth=1) for i in range(8)]
    ys = [ObjectFifo(typ(1024), name=f"y{i}", depth=1) for i in range(8)]

    def core(x, w, p, f, z):
        xv, pv = x.acquire(1), p.acquire(1)
        z(pv)
        for row in range_(16):
            for col in range_(8):
                wv = w.acquire(1)
                f(xv, wv, pv, row, col)
                w.release(1)
        x.release(1)
        p.release(1)

    def finish(p, r0, r1, y, f, index):
        pv, a, b, yv = p.acquire(1), r0.acquire(1), r1.acquire(1), y.acquire(1)
        f(pv, a, b, yv, index)
        p.release(1)
        r0.release(1)
        r1.release(1)
        y.release(1)

    workers = [Worker(core, [xs.cons(), ws[i].cons(), ps[i].prod(), fn, zero],
                      stack_size=12288) for i in range(8)]
    workers += [Worker(finish, [ps[i].cons(), residuals[0].cons(), residuals[1].cons(), ys[i].prod(), add, i],
                       stack_size=1024) for i in range(8)]

    def seq(x, w, r0, r1, y, hx, hw, hr, hy):
        hx.fill(x, tap=TAP((stride + 2048,), 0, [1, 1, 2, 2048], [0, 0, stride, 1]))
        hr[0].fill(r0)
        hr[1].fill(r1)
        for i in range(8):
            hw[i].fill(w, tap=TAP((4194304,), i * 524288, [1, 1, 1, 524288], [0, 0, 0, 1]))
            hy[i].drain(y, tap=TAP((8192,), i * 256, [1, 2, 2, 256], [0, 4096, 2048, 1]), wait=True)

    return Program(iron.get_current_device(), Runtime(seq, [
        typ(stride + 2048), wt(4194304), typ(2048), typ(2048), typ(8192), xs.prod(),
        [f.prod() for f in ws], [f.prod() for f in residuals], [f.cons() for f in ys],
    ]), workers=workers).resolve_program()

if __name__ == "__main__":
    for stride in (2048, 55296, 61440):
        path = KERNEL_ROOT / f"bf16-prefill-output-b2-s{stride}"
        path.mkdir(parents=True, exist_ok=True)
        design.specialize(stride=stride).compile(path / "design.xclbin", path / "instructions.bin")
        (path / "config.json").write_text(json.dumps(dict(
            schema_version=1, channels=2048, batch=2, cores=16, input_stride=stride,
            dtype="bfloat16", residual_layout="separate_tokens",
            output_layout="token_projection_residual")) + "\n")
        print(path, flush=True)

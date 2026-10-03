# SPDX-License-Identifier: Apache-2.0
"""Batched BF16 projection primitive for NPU prefill; no runtime dispatch."""

import json
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_common import KERNEL_ROOT, typ, external


@iron.jit
def design(x: In, w: In, y: Out, *, k: CompileTime[int], rows: CompileTime[int],
           batch: CompileTime[int]):
    if (k, rows) not in ((2048, 8192), (8192, 2048)) or batch not in (1, 2):
        raise ValueError("Supported: FFN key/value shapes with batch 1 or 2")
    bt = lambda n: np.ndarray[(n,), np.dtype[bfloat16]]
    fn = external(
        "rwkv7_prefill_projection_tile", "prefill_projection_bf16.cc",
        [bt(batch * k), bt(4096), typ(batch * 16), np.int32],
        optimization="-O3",
        extra_compile_flags=(f"-DPREFILL_BATCH={batch}", f"-DPREFILL_K={k}"),
    )
    zero = fn.object_file.bind("rwkv7_prefill_projection_zero", [typ(batch * 16)])
    xs = [ObjectFifo(bt(batch * k), name=f"x{i}", depth=1) for i in range(8)]
    ws = [ObjectFifo(bt(4096), name=f"w{i}", depth=2) for i in range(8)]
    ys = [ObjectFifo(typ(batch * 16), name=f"y{i}", depth=2) for i in range(8)]

    def core(x, w, y, f, z):
        xv = x.acquire(1)
        for _ in range_(rows // (8 * 16)):
            out = y.acquire(1)
            z(out)
            for col in range_(k // 256):
                wv = w.acquire(1)
                f(xv, wv, out, col)
                w.release(1)
            y.release(1)
        x.release(1)

    workers = [Worker(core, [xs[i].cons(), ws[i].cons(), ys[i].prod(), fn, zero],
                      stack_size=12288) for i in range(8)]

    def seq(x, w, y, hx, hw, hy):
        for i in range(8):
            hx[i].fill(x)
            hw[i].fill(w, tap=TAP((rows * k,), i * rows * k // 8,
                                 [1, 1, 1, rows * k // 8], [0, 0, 0, 1]))
            # Per-core stream: output tile, token, 16 channels. Scatter into
            # conventional token-major FP32 rows for downstream NPU stages.
            hy[i].drain(y, tap=TAP((batch * rows,), i * rows // 8,
                                  [rows // 128, batch, 1, 16],
                                  [16, rows, 0, 1]), wait=True)

    return Program(iron.get_current_device(), Runtime(seq, [
        bt(batch * k), bt(rows * k), typ(batch * rows),
        [f.prod() for f in xs], [f.prod() for f in ws], [f.cons() for f in ys],
    ]), workers=workers).resolve_program()


if __name__ == "__main__":
    for k, rows in ((2048, 8192), (8192, 2048)):
        for batch in (1, 2):
            path = KERNEL_ROOT / f"bf16-prefill-projection-{k}-{rows}-b{batch}"
            path.mkdir(parents=True, exist_ok=True)
            design.specialize(k=k, rows=rows, batch=batch).compile(
                path / "design.xclbin", path / "instructions.bin")
            (path / "config.json").write_text(json.dumps(dict(
                schema_version=1, k=k, rows=rows, batch=batch, cores=8,
                input_dtype="bfloat16", weight_dtype="bfloat16",
                output_dtype="float32", layout="token_major",
                weight_layout="output_tiles_16_k_tiles_256",
                weight_bytes_per_batch=rows * k * 2,
            )) + "\n")
            print(path, flush=True)

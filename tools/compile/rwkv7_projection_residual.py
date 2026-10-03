# SPDX-License-Identifier: Apache-2.0
"""BF16 attention output projection and FP32 residual in one program."""

import json
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_common import KERNEL_ROOT, typ, external


def projection_residual_graph():
    wt = lambda n: np.ndarray[(n,), np.dtype[bfloat16]]
    tile = 4096
    fn = external(
        "rwkv7_ffn_value_tile",
        "ffn_pipeline_bf16.cc",
        [typ(2048), wt(tile), typ(256), np.int32, np.int32],
        optimization="-O3",
    )
    zero = fn.object_file.bind("rwkv7_ffn_zero256", [typ(256)])
    copy = fn.object_file.bind("rwkv7_ffn_copy1024", [typ(1024), typ(1024)])
    add = fn.object_file.bind("rwkv7_ffn_residual", [typ(4096), typ(4096)])
    xfifo = ObjectFifo(typ(2048), name="input", depth=1)
    ws = [ObjectFifo(wt(tile), name=f"w{i}", depth=2) for i in range(8)]
    groups = [ObjectFifo(typ(1024), name=f"group{i}", depth=1) for i in range(2)]
    ys = []
    for group in groups:
        ys += group.prod().join([0, 256, 512, 768], obj_types=[typ(256)] * 4)
    pair = ObjectFifo(typ(4096), name="pair", depth=1)
    joins = pair.prod().join(
        [0, 1024, 2048], obj_types=[typ(1024), typ(1024), typ(2048)]
    )
    final = ObjectFifo(typ(4096), name="final", depth=1)

    def core(x, w, y, f, z):
        xv, yv = (x.acquire(1), y.acquire(1))
        z(yv)
        for row in range_(16):
            for col in range_(8):
                wv = w.acquire(1)
                f(xv, wv, yv, row, col)
                w.release(1)
        x.release(1)
        y.release(1)

    def unary(x, y, f):
        xv, yv = (x.acquire(1), y.acquire(1))
        f(xv, yv)
        x.release(1)
        y.release(1)

    workers = [
        Worker(
            core, [xfifo.cons(), ws[i].cons(), ys[i].prod(), fn, zero], stack_size=12288
        )
        for i in range(8)
    ]
    workers += [
        Worker(unary, [groups[i].cons(), joins[i].prod(), copy], stack_size=12288)
        for i in range(2)
    ]
    workers += [Worker(unary, [pair.cons(), final.prod(), add], stack_size=12288)]
    return (workers, xfifo, ws, joins[2], final)


def projection_residual_program():
    workers, xfifo, ws, residual_input, final = projection_residual_graph()
    wt = lambda n: np.ndarray[(n,), np.dtype[bfloat16]]
    weights = 4096 * 1024

    def seq(x, w, res, out, hx, hw, hr, ho):
        hx.fill(x)
        for i in range(8):
            hw[i].fill(
                w,
                tap=TAP(
                    (weights,),
                    i * (weights // 8),
                    [1, 1, 1, weights // 8],
                    [0, 0, 0, 1],
                ),
            )
        hr.fill(res)
        ho.drain(out, wait=True)

    return Program(
        iron.get_current_device(),
        Runtime(
            seq,
            [
                typ(2048),
                wt(weights),
                typ(2048),
                typ(4096),
                xfifo.prod(),
                [f.prod() for f in ws],
                residual_input.prod(),
                final.cons(),
            ],
        ),
        workers=workers,
    ).resolve_program()


@iron.jit
def design(x: In, w: In, residual: In, result: Out):
    return projection_residual_program()


if __name__ == "__main__":
    path = KERNEL_ROOT / "bf16-projection-residual"
    path.mkdir(parents=True, exist_ok=True)
    design.compile(path / "design.xclbin", path / "instructions.bin")
    (path / "config.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                dtype="bfloat16",
                channels=2048,
                cores=11,
                exact_fp32=False,
            )
        )
        + "\n"
    )
    print(path, flush=True)

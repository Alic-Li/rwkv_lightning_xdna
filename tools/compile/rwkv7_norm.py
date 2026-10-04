# SPDX-License-Identifier: Apache-2.0
"""Pinned upstream FP32 LayerNorm template with FP32 affine output."""

import json
import os
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Worker, Runtime, Program
from rwkv7_common import C, ROOT, KERNEL_ROOT, typ, external

os.environ["MLIR_AIE_KERNEL_SOURCES"] = str(ROOT / "third_party/mlir-aie")


@iron.jit
def design(x: In, w: In, b: In, y: Out):
    p = ObjectFifo(typ((3 * C)), name="packed", depth=1)
    inputs = p.prod().join([0, C, (2 * C)], obj_types=[typ(C)] * 3)
    o = ObjectFifo(typ(C), name="output", depth=1)
    fn = external("rwkv7_layer_norm", "norm_fp32.cc", [typ((3 * C)), typ(C)])

    def core(p, o, f):
        a, b = p.acquire(1), o.acquire(1)
        f(a, b)
        p.release(1)
        o.release(1)

    worker = Worker(core, [p.cons(), o.prod(), fn], stack_size=12288)

    def seq(x, w, b, y, hi, ho):
        for f, v in zip(hi, [x, w, b]):
            f.fill(v)
        ho.drain(y, wait=True)

    return Program(
        iron.get_current_device(),
        Runtime(seq, [typ(C)] * 4 + [[f.prod() for f in inputs], o.cons()]),
        workers=[worker],
    ).resolve_program()


if __name__ == "__main__":
    p = KERNEL_ROOT / "upstream-norm"
    p.mkdir(parents=True, exist_ok=True)
    design.compile(p / "design.xclbin", p / "instructions.bin")
    (p / "config.json").write_text(
        json.dumps(dict(schema_version=1, dtype="float32", channels=C, epsilon=1e-5))
        + "\n"
    )

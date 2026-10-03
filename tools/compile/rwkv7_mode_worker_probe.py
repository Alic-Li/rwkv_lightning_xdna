# SPDX-License-Identifier: Apache-2.0
"""Offline diagnostic: LayerNorm/projection opcodes on the same eight dot workers.

ABI: packed input is x[2048], gamma[2048], beta[2048], opcode, padding[15].
Opcode 0 uses norm instructions and zero residual; opcode 1 uses projection
instructions and packed BF16 weights. Both produce [result, result+residual].
This is a diagnostic format, not the model's production artifact ABI.
"""

import json
from pathlib import Path
from aie.dialects import arith, memref
from aie.ir import IndexType, F32Type
from aie.helpers.dialects.scf import if_, else_
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_common import KERNEL_ROOT, typ, external


def graph():
    wt = lambda n: np.ndarray[(n,), np.dtype[bfloat16]]
    tile = 4096
    fn = external(
        "rwkv7_ffn_value_tile",
        "mode_projection_probe.cc",
        [typ(6160), wt(tile), typ(256), np.int32, np.int32],
        optimization="-O3",
        stack_size=16384,
    )
    zero = fn.object_file.bind("rwkv7_ffn_zero256", [typ(256)])
    copy = fn.object_file.bind("rwkv7_ffn_copy1024", [typ(1024), typ(1024)])
    add = fn.object_file.bind("rwkv7_ffn_residual", [typ(4096), typ(4096)])
    xfifo = ObjectFifo(typ(6160), name="input", depth=1)
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

    normalize = fn.object_file.bind(
        "rwkv7_mode_norm_slice", [typ(6160), typ(256), np.int32]
    )

    def core(x, w, y, f, z, n, lane):
        xv, yv = x.acquire(1), y.acquire(1)
        opcode = memref.LoadOp(xv, [arith.ConstantOp(IndexType.get(), 6144)]).result
        condition = arith.CmpFOp(
            arith.CmpFPredicate.OEQ, opcode, arith.ConstantOp(F32Type.get(), 0.0)
        ).result
        with if_(condition) as branch:
            n(xv, yv, lane)
        with else_(branch):
            z(yv)
            for row in range_(16):
                for col in range_(8):
                    wv = w.acquire(1)
                    f(xv, wv, yv, row, col)
                    w.release(1)
        x.release(1)
        y.release(1)

    def unary(x, y, f):
        xv, yv = x.acquire(1), y.acquire(1)
        f(xv, yv)
        x.release(1)
        y.release(1)

    workers = [
        Worker(
            core,
            [xfifo.cons(), ws[i].cons(), ys[i].prod(), fn, zero, normalize, i],
            stack_size=20480,
            dynamic_objfifo_lowering=True,
        )
        for i in range(8)
    ]
    workers += [
        Worker(unary, [groups[i].cons(), joins[i].prod(), copy], stack_size=12288)
        for i in range(2)
    ]
    workers += [Worker(unary, [pair.cons(), final.prod(), add], stack_size=12288)]

    return workers, xfifo, ws, joins[2], final


def program(norm_only):
    workers, x, ws, res, y = graph()
    raw = lambda n: np.ndarray[(n,), np.dtype[np.uint8]]

    def seq(a, b, r, o, hx, hw, hr, hy):
        hx.fill(a)
        if not norm_only:
            for i in range(8):
                hw[i].fill(
                    b,
                    tap=TAP((8388608,), i * 1048576, [1, 1, 1, 1048576], [0, 0, 0, 1]),
                )
        hr.fill(r)
        hy.drain(o, wait=True)

    # Bind all static shim endpoints even for the norm instruction variant.
    return Program(
        iron.get_current_device(),
        Runtime(
            seq,
            [
                typ(6160),
                raw(8388608),
                typ(2048),
                typ(4096),
                x.prod(),
                [w.prod() for w in ws],
                res.prod(),
                y.cons(),
            ],
        ),
        workers=workers,
    ).resolve_program()


@iron.jit
def norm(a: In, b: In, r: In, o: Out):
    return program(True)


@iron.jit
def project(a: In, b: In, r: In, o: Out):
    return program(False)


if __name__ == "__main__":
    import argparse
    import shutil
    import hashlib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.resolve() == KERNEL_ROOT.resolve():
        parser.error("Use a separate diagnostic artifact root")
    for name, design in [("project", project), ("norm", norm)]:
        path = args.output_root / name
        path.mkdir(parents=True, exist_ok=True)
        design.compile(path / "design.xclbin", path / "instructions.bin")
    p, n = args.output_root / "project", args.output_root / "norm"
    pb, nb = (p / "design.prj/main.pdi").read_bytes(), (
        n / "design.prj/main.pdi"
    ).read_bytes()
    if pb != nb:
        raise RuntimeError("Different device programs: refusing to share an xclbin")
    shutil.copyfile(p / "design.xclbin", n / "design.xclbin")
    digest = hashlib.sha256(pb).hexdigest()
    for name, opcode in [("project", 1), ("norm", 0)]:
        metadata = dict(
            schema_version=1,
            diagnostic_only=True,
            cores=11,
            channels=2048,
            pdi_sha256=digest,
            shared_program="project",
            precision="bf16",
            opcode=opcode,
            packed_input_floats=6160,
        )
        (args.output_root / name / "config.json").write_text(
            json.dumps(metadata) + "\n"
        )
    print("shared PDI", digest)

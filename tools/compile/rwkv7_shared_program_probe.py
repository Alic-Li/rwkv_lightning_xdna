# SPDX-License-Identifier: Apache-2.0
"""Offline diagnostic: co-resident norm and projection with separate instructions.

Run rwkv-working-set-bench on the output root to validate and measure it.
This is not a production model artifact generator.
"""

import json
import re
from pathlib import Path
import numpy as np
import aie.iron as iron
from aie.ir import Module
from aie.iron import In, Out, ObjectFifo, Worker, Runtime, Program
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_common import KERNEL_ROOT, typ, external
from rwkv7_projection_residual import projection_residual_graph


def shared_program(quantized=False, norm_only=False):
    workers, xfifo, ws, residual_input, final = projection_residual_graph(quantized)
    weights = (4160 if quantized else 4096) * 1024

    def unary(x, y, f):
        xv, yv = x.acquire(1), y.acquire(1)
        f(xv, yv)
        x.release(1)
        y.release(1)

    packed = ObjectFifo(typ(6144), name="norm_packed", depth=1)
    norm_inputs = packed.prod().join([0, 2048, 4096], obj_types=[typ(2048)] * 3)
    norm_output = ObjectFifo(typ(2048), name="norm_output", depth=1)
    norm_fn = external("rwkv7_layer_norm", "norm_fp32.cc", [typ(6144), typ(2048)])
    workers.append(
        Worker(unary, [packed.cons(), norm_output.prod(), norm_fn], stack_size=12288)
    )
    # Runtime arguments are byte-addressed; FIFO element types and all device
    # arithmetic remain unchanged. Norm touches only the first 8192 bytes of
    # each BO, even where the shared signature declares a larger address span.
    raw = lambda n: np.ndarray[(n,), np.dtype[np.uint8]]
    nbytes = weights if quantized else weights * 2

    def tap(total, count, offset=0):
        return TAP((total,), offset, [1, 1, 1, count], [0, 0, 0, 1])

    def seq(x, w, res, out, hx, hw, hr, ho, hn, hno):
        hx.fill(x, tap=tap(8192, 8192))
        for i in range(8):
            hw[i].fill(w, tap=tap(nbytes, nbytes // 8, i * (nbytes // 8)))
        hr.fill(res, tap=tap(8192, 8192))
        ho.drain(out, tap=tap(16384, 16384), wait=True)
        for f, v, size in zip(hn, [x, w, res], [8192, nbytes, 8192]):
            f.fill(v, tap=tap(size, 8192))
        hno.drain(out, tap=tap(16384, 8192), wait=True)

    module = Program(
        iron.get_current_device(),
        Runtime(
            seq,
            [
                raw(8192),
                raw(nbytes),
                raw(8192),
                raw(16384),
                xfifo.prod(),
                [f.prod() for f in ws],
                residual_input.prod(),
                final.cons(),
                [f.prod() for f in norm_inputs],
                norm_output.cons(),
            ],
        ),
        workers=workers,
    ).resolve_program()
    # Resolve both DMA endpoints before selecting the runtime instructions. This
    # preserves identical placement and static FIFOs in the two compilations.
    # Match only this generator's known task syntax; fail if its shape changes.
    source = str(module)
    pattern = r"      (%[0-9]+) = aiex\.dma_configure_task_for @([^ ]+) \{.*?^      \}[^\n]*\n"
    removed = set()

    def prune(match):
        if match.group(2).startswith("norm_") != norm_only:
            removed.add(match.group(1))
            return ""
        return match.group(0)

    source = re.sub(pattern, prune, source, flags=re.S | re.M)
    source = "\n".join(
        line
        for line in source.splitlines()
        if not any(
            re.search(
                r"aiex\.dma_(?:start|await|free)_task\(" + re.escape(task) + r"\)", line
            )
            for task in removed
        )
    )
    if len(removed) != (11 if norm_only else 4):
        raise RuntimeError(f"Unexpected runtime DMA tasks: {removed}")
    module = Module.parse(source)
    if not module.operation.verify():
        raise RuntimeError("Selected runtime sequence failed MLIR verification")
    return module


@iron.jit
def project_bf16(x: In, w: In, residual: In, result: Out):
    return shared_program()


@iron.jit
def norm_bf16(x: In, w: In, residual: In, result: Out):
    return shared_program(norm_only=True)


@iron.jit
def project_int8(x: In, w: In, residual: In, result: Out):
    return shared_program(quantized=True)


@iron.jit
def norm_int8(x: In, w: In, residual: In, result: Out):
    return shared_program(quantized=True, norm_only=True)


if __name__ == "__main__":
    import argparse
    import hashlib
    import shutil

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--int8", action="store_true")
    args = parser.parse_args()
    if args.output_root.resolve() == KERNEL_ROOT.resolve():
        parser.error("Use a separate diagnostic artifact root")
    name = "int8-projection-residual" if args.int8 else "bf16-projection-residual"
    designs = [
        (name, project_int8 if args.int8 else project_bf16),
        ("upstream-norm", norm_int8 if args.int8 else norm_bf16),
    ]
    for stage, design in designs:
        path = args.output_root / stage
        path.mkdir(parents=True, exist_ok=True)
        design.compile(path / "design.xclbin", path / "instructions.bin")
    p, n = args.output_root / name, args.output_root / "upstream-norm"
    pb, nb = (p / "design.prj/main.pdi").read_bytes(), (
        n / "design.prj/main.pdi"
    ).read_bytes()
    if pb != nb:
        raise RuntimeError("Different device programs: refusing to share an xclbin")
    digest = hashlib.sha256(pb).hexdigest()
    shutil.copyfile(p / "design.xclbin", n / "design.xclbin")
    for stage, _ in designs:
        metadata = dict(
            schema_version=1,
            diagnostic_only=True,
            cores=12,
            channels=2048,
            pdi_sha256=digest,
            shared_program=name,
            precision="int8" if args.int8 else "bf16",
        )
        (args.output_root / stage / "config.json").write_text(
            json.dumps(metadata) + "\n"
        )
    print("shared PDI", digest, flush=True)

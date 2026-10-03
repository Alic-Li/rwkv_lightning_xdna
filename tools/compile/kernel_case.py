# SPDX-License-Identifier: Apache-2.0
"""Offline compiler and fixture exporter. This module never dispatches an NPU."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
TEST_KERNEL_ROOT = Path(os.environ.get("RWKV_XDNA_TEST_KERNEL_DIR", ROOT / "build/kernels/test"))
os.environ["MLIR_AIE_KERNEL_SOURCES"] = str(ROOT / "third_party/mlir-aie")
sys.path.insert(0, str(ROOT / "third_party/kernel_tests"))

import numpy as np
from aie.iron import kernels
from aie.iron.algorithms import kernel_design as kd
from aie.iron.device import NPU2
from aie.utils import set_current_device

set_current_device(NPU2())
from cases import inputs_for
from kernel_cases import CASES


def case_id(case):
    if case.factory == "cascade_mm":
        return "cascade-mm"
    return case.factory + "-" + hashlib.sha256(case.name.encode()).hexdigest()[:12]


def selected(tier):
    supported = [c for c in CASES if not c.devices or "npu2" in c.devices]
    if tier == "all":
        return supported
    result = [c for c in supported if c.smoke]
    missing = {c.factory for c in supported} - {c.factory for c in result}
    result += [c for c in supported if c.factory in missing]
    return result


def find_case(identifier):
    return next(c for c in selected("all") if case_id(c) == identifier)


def compile_case(case):
    directory = TEST_KERNEL_ROOT / case_id(case)
    directory.mkdir(parents=True, exist_ok=True)
    fn = case.fn()
    inputs = inputs_for(case, "random", np.random.default_rng(1000))
    design = kd.design(
        getattr(kernels, case.factory),
        **case.harness_opts(),
        params=fn.param_values(inputs),
        guard=True,
        **case.kwargs,
    )
    design.compile(directory / "design.xclbin", directory / "instructions.bin")
    physical_inputs = iter(kd.host_layout(fn, inputs))
    buffers = []
    for index, arg in enumerate(kd.host_args(fn, calls=case.calls, guard=True)):
        output = arg.direction.__name__ == "Out"
        name = f"{'output' if output else 'input'}-{index}.bin"
        size = arg.n_elements * np.dtype(arg.dtype).itemsize
        if not output:
            array = np.ascontiguousarray(next(physical_inputs))
            if array.nbytes != size:
                raise ValueError("Physical input size mismatch")
            array.tofile(directory / name)
        buffers.append(dict(direction="out" if output else "in", bytes=size, file=name))
    manifest = dict(
        schema_version=1,
        case=case.name,
        case_id=case_id(case),
        xclbin="design.xclbin",
        instructions="instructions.bin",
        kernel="MLIR_AIE",
        repetitions=3,
        buffers=buffers,
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case")
    parser.add_argument("--list", choices=["smoke", "all"])
    args = parser.parse_args()
    if args.list:
        print(
            json.dumps(
                [
                    {"id": case_id(c), "name": c.name, "factory": c.factory}
                    for c in selected(args.list)
                ],
                indent=2,
            )
        )
    elif args.case:
        print(compile_case(find_case(args.case)))
    else:
        parser.error("provide --case or --list")

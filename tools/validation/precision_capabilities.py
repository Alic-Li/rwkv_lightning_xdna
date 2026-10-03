#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline AIE2P precision probes, with BF16/INT8 positive controls.

Compilation support is not a hardware dispatch or a throughput measurement.
This script never changes the production kernels or selects a fallback dtype.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sysconfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("build/precision-capability")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    packages = Path(sysconfig.get_path("purelib"))
    clang = packages / "llvm-aie/bin/clang++"
    include = packages / "mlir_aie/include"
    target = "aie2p-none-unknown-elf"
    version = subprocess.run(
        [str(clang), "--version"], check=True, capture_output=True, text=True
    ).stdout
    macros = subprocess.run(
        [str(clang), f"--target={target}", "-std=c++20", "-dM", "-E", "-x", "c++", "-"],
        input="",
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (args.output / "macros.txt").write_text(macros)
    results = {}
    for name, dtype, accumulator in [
        ("bf16", "bfloat16", "float"),
        ("int8", "int8_t", "int32_t"),
        ("fp16", "_Float16", "float"),
    ]:
        source = args.output / f"{name}.cc"
        source.write_text(
            "#include <aie_api/aie.hpp>\n"
            f'extern "C" void probe(const {dtype} *x, const {dtype} *y, {accumulator} *z) {{\n'
            "  aie::store_v(z, aie::mul(aie::load_v<32>(x), aie::load_v<32>(y))"
            f".to_vector<{accumulator}>());\n}}\n"
        )
        obj = args.output / f"{name}.o"
        obj.unlink(missing_ok=True)
        command = [
            str(clang),
            f"--target={target}",
            "-std=c++20",
            "-O2",
            # Only vector arithmetic is probed. Exclude optional ADF
            # graph wrappers, whose adf.h is absent in the wheel.
            "-D__AIE_API_AIE_ADF_HPP__",
            f"-I{include}",
            "-c",
            str(source),
            "-o",
            str(obj),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        (args.output / f"{name}.log").write_text(result.stdout + result.stderr)
        results[name] = {
            "compiled": result.returncode == 0,
            "exit_code": result.returncode,
            "command": command,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "object_sha256": (
                hashlib.sha256(obj.read_bytes()).hexdigest() if obj.exists() else None
            ),
            "errors": [line for line in result.stderr.splitlines() if "error:" in line],
        }
    controls_passed = results["bf16"]["compiled"] and results["int8"]["compiled"]
    report = {
        "scope": "Installed compiler/API support for vector multiply; no NPU dispatch or native throughput claim",
        "target": target,
        "compiler": version.strip(),
        "architecture_macros": [
            line
            for line in macros.splitlines()
            if line.startswith(("#define __AIE_ARCH", "#define __AIE_MODEL"))
        ],
        "positive_controls_passed": controls_passed,
        "probes": results,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if controls_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: Apache-2.0
"""Compile the FP32 resident decode pipeline (offline, no NPU dispatch)."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "--skip-base", action="store_true", help="reuse existing rwkv7_full artifacts"
)
p.add_argument(
    "--native-fp32",
    action="store_true",
    help="experimental faster arithmetic; does not meet the legacy real-model pointwise tolerance",
)
a = p.parse_args()
env = dict(
    os.environ,
    MLIR_AIE_KERNEL_SOURCES=str(ROOT / "third_party/mlir-aie"),
    RWKV_XDNA_EXACT="0" if a.native_fp32 else "1",
)
commands = [] if a.skip_base else [("rwkv7_full.py", [])]
commands += [
    ("rwkv7_resident.py", []),
    ("rwkv7_array.py", []),
    ("rwkv7_array.py", ["--k", "2048", "--rows", "8192"]),
    ("rwkv7_array32.py", []),
    ("rwkv7_array32.py", ["--k", "2048", "--rows", "8192"]),
    ("rwkv7_heads.py", []),
    ("rwkv7_mix.py", []),
    ("rwkv7_rank.py", []),
    ("rwkv7_stages.py", []),
    ("rwkv7_value.py", []),
    ("rwkv7_ffn.py", []),
    ("rwkv7_norm.py", []),
]
for script, args in commands:
    subprocess.run(
        [sys.executable, str(ROOT / "tools/compile" / script), *args],
        cwd=ROOT,
        env=env,
        check=True,
    )
print(
    "Compilation complete; run rwkv-array-test, rwkv-graph-test, and real-model alignment before accepting artifacts."
)

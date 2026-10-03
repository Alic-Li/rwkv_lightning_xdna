# SPDX-License-Identifier: Apache-2.0
"""Compile the single supported BF16/FP32 resident pipeline, without NPU dispatch."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/kernels/rwkv7-bf16")
    args = parser.parse_args()
    env = dict(os.environ, MLIR_AIE_KERNEL_SOURCES=str(ROOT / "third_party/mlir-aie"),
               RWKV_XDNA_KERNEL_DIR=str(args.output.resolve()))
    # Ordered only for reproducible build logs; each program has its own directory.
    for stage in ("norm", "norm_mix", "attention_projections",
                  "value", "recurrence_stage", "projection_residual", "channel_mix", "head"):
        subprocess.run([sys.executable, str(ROOT / "tools/compile" / f"rwkv7_{stage}.py")],
                       cwd=ROOT, env=env, check=True)
    print("Compiled BF16 pipeline. Validate with C++ dispatch, numerical and guard checks.")

if __name__ == "__main__":
    main()

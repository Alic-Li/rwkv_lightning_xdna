# SPDX-License-Identifier: Apache-2.0
"""Compile BF16/FP32 resident programs and optional W8A16 FFN, without dispatch."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/kernels/rwkv7-bf16")
    parser.add_argument("--int8-ffn", action="store_true", help="Also compile experimental W8A16 ChannelMix")
    parser.add_argument("--int8-ffn-output", action="store_true", help="Also compile W8A16 ChannelMix and attention output")
    parser.add_argument("--prefill-batch2", action="store_true", help="Also compile two-token ChannelMix prefill for the selected precisions")
    args = parser.parse_args()
    env = dict(os.environ, MLIR_AIE_KERNEL_SOURCES=str(ROOT / "third_party/mlir-aie"),
               RWKV_XDNA_KERNEL_DIR=str(args.output.resolve()))
    # Ordered only for reproducible build logs; each program has its own directory.
    for stage in ("norm", "norm_mix", "attention_projections",
                  "value_recurrence", "recurrence_stage", "projection_residual", "channel_mix", "head"):
        subprocess.run([sys.executable, str(ROOT / "tools/compile" / f"rwkv7_{stage}.py")],
                       cwd=ROOT, env=env, check=True)
    if args.prefill_batch2:
        subprocess.run([sys.executable, str(ROOT / "tools/compile/rwkv7_prefill_output.py")],
                       cwd=ROOT, env=env, check=True)
        subprocess.run([sys.executable, str(ROOT / "tools/compile/rwkv7_attention_projections.py"), "--batch", "2"],
                       cwd=ROOT, env=env, check=True)
        subprocess.run([sys.executable, str(ROOT / "tools/compile/rwkv7_prefill_channel_mix.py")],
                       cwd=ROOT, env=env, check=True)
    if args.prefill_batch2 and (args.int8_ffn or args.int8_ffn_output):
        subprocess.run([sys.executable, str(ROOT / "tools/compile/rwkv7_prefill_channel_mix.py"), "--int8"],
                       cwd=ROOT, env=env, check=True)
    if args.int8_ffn or args.int8_ffn_output:
        subprocess.run([sys.executable, str(ROOT / "tools/compile/rwkv7_channel_mix_int8.py")],
                       cwd=ROOT, env=env, check=True)
    if args.prefill_batch2 and args.int8_ffn_output:
        subprocess.run([sys.executable, str(ROOT / "tools/compile/rwkv7_prefill_output.py"), "--int8"],
                       cwd=ROOT, env=env, check=True)
    if args.int8_ffn_output:
        subprocess.run([sys.executable, str(ROOT / "tools/compile/rwkv7_projection_residual_int8.py")],
                       cwd=ROOT, env=env, check=True)
    print("Compiled BF16 pipeline. Validate with C++ dispatch, numerical and guard checks.")

if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Compile resident BF16/FP32 programs and optional INT8 FFN; never dispatch."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def compile_stages(args):
    """One explicit manifest shared by decode and the optional prefill plans."""
    stages = [
        (name, [])
        for name in (
            "norm",
            "norm_mix",
            "attention_projections",
            "value_recurrence",
            "recurrence_stage",
            "projection_residual",
            "channel_mix",
            "head",
            "decode_recurrence_projection",
        )
    ]
    if args.int8_ffn:
        stages.append(("channel_mix_int8", []))
    if args.prefill_batch2 or args.prefill_chunk4:
        stages.extend(
            [
                ("prefill_recurrence_projection", []),
                ("prefill_recurrence", []),
                ("prefill_output", ["--share-program"] if args.prefill_chunk4 else []),
                ("attention_projections", ["--batch", "2"]),
            ]
        )
        stages.extend(
            [
                ("prefill_channel_mix", []),
                ("prefill_channel_mix", ["--recurrence-input"]),
            ]
        )
        if args.prefill_chunk4:
            stages.append(("prefill_chunk4", ["--projection-input"]))
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/kernels/rwkv7-bf16"
    )
    parser.add_argument(
        "--int8-ffn", action="store_true", help="Also compile the W8A8 streaming FFN"
    )
    prefill = parser.add_mutually_exclusive_group()
    prefill.add_argument("--prefill-batch2", action="store_true")
    prefill.add_argument(
        "--prefill-chunk4", action="store_true", help="Includes batch2 artifacts"
    )
    args = parser.parse_args()
    env = dict(
        os.environ,
        MLIR_AIE_KERNEL_SOURCES=str(ROOT / "third_party/mlir-aie"),
        RWKV_XDNA_KERNEL_DIR=str(args.output.resolve()),
    )
    for stage, flags in compile_stages(args):
        subprocess.run(
            [sys.executable, str(ROOT / "tools/compile" / f"rwkv7_{stage}.py"), *flags],
            cwd=ROOT,
            env=env,
            check=True,
        )
    print(
        "Compiled resident pipeline. Validate with C++ dispatch, numerical and guard checks."
    )


if __name__ == "__main__":
    main()

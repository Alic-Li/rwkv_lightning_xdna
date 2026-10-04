# SPDX-License-Identifier: Apache-2.0
"""Shared offline compiler setup for the BF16 production pipeline."""

import os
from pathlib import Path
import numpy as np
from aie.iron import ExternalFunction
from aie.iron.device import NPU2
from aie.utils import set_current_device

ROOT = Path(__file__).resolve().parents[2]
KERNEL_ROOT = Path(
    os.environ.get("RWKV_XDNA_KERNEL_DIR", ROOT / "build/kernels/rwkv7-bf16")
)
C = int(os.environ.get("RWKV_XDNA_CHANNELS", "2048"))
H = int(os.environ.get("RWKV_XDNA_HIDDEN", "8192"))
V = int(os.environ.get("RWKV_XDNA_VOCAB", "65536"))
HEADS = C // 64
set_current_device(NPU2())


def typ(n):
    return np.ndarray[(n,), np.dtype[np.float32]]


def external(
    name, source, types, optimization="-Oz", stack_size=8192, extra_compile_flags=()
):
    return ExternalFunction(
        name,
        source_file=str(ROOT / "kernels/rwkv" / source),
        arg_types=types,
        compile_flags=[
            optimization,
            f"-DRWKV_CHANNELS={C}",
            f"-DRWKV_HIDDEN={H}",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-D__AIE_API_FP32_EMULATION__=1",
            *extra_compile_flags,
        ],
        stack_size_override=stack_size,
    )

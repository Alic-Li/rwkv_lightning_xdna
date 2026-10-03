# SPDX-License-Identifier: Apache-2.0
"""Experimental W8A16 attention output projection with fused FP32 residual."""
import json
import aie.iron as iron
from aie.iron import In, Out
from rwkv7_projection_residual import projection_residual_program
from rwkv7_common import KERNEL_ROOT


@iron.jit
def design(x: In, w: In, residual: In, result: Out):
    return projection_residual_program(quantized=True)


if __name__ == "__main__":
    path = KERNEL_ROOT / "int8-projection-residual"
    path.mkdir(parents=True, exist_ok=True)
    design.compile(path / "design.xclbin", path / "instructions.bin")
    (path / "config.json").write_text(json.dumps(dict(
        schema_version=1, dtype="int8", channels=2048, cores=11,
        tile_bytes=4160, quantization="symmetric_per_output_127",
        scale="fp16_expanded_fp32", activation_dtype="bfloat16",
        accumulator_dtype="float32")) + "\n")
    print(path, flush=True)

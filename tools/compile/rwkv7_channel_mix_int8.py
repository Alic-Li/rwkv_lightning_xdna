# SPDX-License-Identifier: Apache-2.0
"""W8A16 ChannelMix: streamed signed weights with a fused row-scale epilogue."""

import json
import aie.iron as iron
from aie.iron import In, InOut, Out
from rwkv7_channel_mix import channel_mix_program, TRACE_BYTES, TRACE_ACTIVATION
from rwkv7_common import KERNEL_ROOT


@iron.jit
def design(x: In, parameters: In, weights: In, diagnostic: InOut, result: Out):
    return channel_mix_program(quantized=True)


if __name__ == "__main__":
    path = KERNEL_ROOT / "int8-channel-mix"
    path.mkdir(parents=True, exist_ok=True)
    design.compile(path / "design.xclbin", path / "instructions.bin")
    (path / "config.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                dtype="int8",
                channels=2048,
                hidden=8192,
                key_cores=4,
                value_cores=4,
                tile_bytes=4160,
                quantization="symmetric_per_output_127",
                scale="fp16_expanded_fp32",
                activation_dtype="bfloat16",
                accumulator_dtype="float32",
                activation_conversion="producer_bf16_nearest_even",
                trace_buffer_bytes=TRACE_BYTES,
                trace_key_region=(
                    "activation" if TRACE_ACTIVATION else "matrix_and_activation"
                ),
            )
        )
        + "\n"
    )
    print(path, flush=True)

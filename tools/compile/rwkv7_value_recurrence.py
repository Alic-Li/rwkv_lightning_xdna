# SPDX-License-Identifier: Apache-2.0
"""Value residual plus prepare, FP32 recurrence and finish in one dispatch."""

import json
import aie.iron as iron
from aie.iron import In, InOut
from rwkv7_common import C, KERNEL_ROOT
from rwkv7_recurrence_stage import recurrence_program


@iron.jit
def design(state: InOut, auxiliary: InOut, first: In):
    return recurrence_program(with_value=True)


if __name__ == "__main__":
    path = KERNEL_ROOT / "fused-value-recurrence-stage"
    path.mkdir(parents=True, exist_ok=True)
    design.compile(path / "design.xclbin", path / "instructions.bin")
    (path / "config.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                dtype="float32",
                channels=C,
                head_size=64,
                arena_vectors=30,
                lanes=7,
                fused_value=True,
                exact_fp32=False,
                kernel_optimization="-Os",
                prepare_optimization="-Oz",
            )
        )
        + "\n"
    )
    print(path, flush=True)

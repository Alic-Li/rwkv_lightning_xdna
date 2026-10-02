# SPDX-License-Identifier: Apache-2.0
"""Compile the upstream two-core cascade matmul and export deterministic fixtures."""

import json
from kernel_case import ROOT, np, kernels
from cascade_design import _cascade_design

directory = ROOT / "build/kernels/cascade-mm"
directory.mkdir(parents=True, exist_ok=True)
_cascade_design.compile(directory / "design.xclbin", directory / "instructions.bin")
fn = kernels.cascade_mm(dim_m=16, dim_k=16, dim_n=16)
limit = fn.input_limit(np.int16)
rng = np.random.default_rng(3)
inputs = [rng.integers(-limit, limit, size=(256,)).astype(np.int16) for _ in range(4)]
buffers = []
for i, array in enumerate(inputs):
    array.tofile(directory / f"input-{i}.bin")
    buffers.append(dict(direction="in", bytes=array.nbytes, file=f"input-{i}.bin"))
expected = (
    inputs[0].astype(np.int64).reshape(16, 16)
    @ inputs[1].astype(np.int64).reshape(16, 16)
    + inputs[2].astype(np.int64).reshape(16, 16)
    @ inputs[3].astype(np.int64).reshape(16, 16)
).astype(np.int16)
expected.tofile(directory / "expected.bin")
buffers.append(dict(direction="out", bytes=512, file="output-4.bin"))
manifest = dict(
    schema_version=1,
    case="cascade_mm/16x16x16/int16/pair",
    case_id="cascade-mm",
    xclbin="design.xclbin",
    instructions="instructions.bin",
    kernel="MLIR_AIE",
    repetitions=3,
    buffers=buffers,
)
(directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(directory)

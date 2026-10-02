# Upstream snapshots

| Directory | Source | Version | License |
|---|---|---|---|
| mlir-aie | Xilinx/mlir-aie wheel's aie_kernels and aie_runtime_lib | 1.4.4.dev53+gd53582d | Apache-2.0 WITH LLVM-exception |
| iron | Local amd/IRON source checkout | 3fc9a6084f934108c5c4982eb6ea39a6fcc55b67 | Apache-2.0 |
| kernel_tests | Xilinx/mlir-aie test/python/npu | d53582d | Apache-2.0 WITH LLVM-exception |
| nlohmann | nlohmann/json single_include | v3.12.0 | MIT |

The actual C++ kernels are distributed in the MLIR-AIE wheel, not in IRON's
Python operator directories. Initial import used shell `cp -a` from:

```text
../IRON/ironenv/lib/python3.12/site-packages/mlir_aie/include/aie_kernels
../IRON/ironenv/lib/python3.12/site-packages/mlir_aie/aie_runtime_lib
../IRON/iron
```

These are ordinary files copied into this Git repository, not symlinks or Git
submodules. Uploading the repository includes the 61 `.cc` kernel source files,
their headers and runtime helpers. The sibling `../IRON` directory is not needed
after import. For example, elementwise addition is implemented in
[`mlir-aie/aie_kernels/eltwise/add.cc`](mlir-aie/aie_kernels/eltwise/add.cc).

The execution path is:

```text
copied C++ sources + offline design builder
    -> Peano / MLIR-AIE compilation -> xclbin + instructions
    -> C++ Session loads artifacts through XRT -> NPU execution
```

Changing a copied source requires recompiling the affected design. `--run-only`
loads existing artifacts and will not pick up a source edit by itself.

All kernel source files, architecture-specific headers and runtime/LUT helpers
were retained. Compilation sets `MLIR_AIE_KERNEL_SOURCES` to this project's
`third_party/mlir-aie`, so it compiles this snapshot. The pinned compiler wheel
still supplies the AIE API/toolchain headers and native compiler binaries.

`kernel_tests/cases.py` and `kernel_cases.py` are unmodified upstream files.
`cascade_design.py` extracts the original two-worker design without its Python
execution test. Its source is `test_kernels_e2e.py` at the same commit. Their
license text is in `mlir-aie/LICENSE`. File digests and versions are recorded in
`SOURCES.json`; `tools/validation/report.py` verifies the copied kernel digests.

Do not update one component silently. Update compiler/kernel snapshots and
test definitions together, preserve notices, regenerate provenance, and rerun
the C++ hardware sweep. Put new application kernels outside this directory.

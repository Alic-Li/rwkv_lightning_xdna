# Development rules

- Runtime and inference code is C++17 under `include/`, `src/`, and `apps/`.
- Python is restricted to offline compilation, test-data/reference experiments,
  and orchestration of C++ test executables. Never dispatch the NPU from Python.
- Use uv and the project-local `.venv`; do not depend on the sibling IRON checkout.
- `third_party/` contains versioned upstream snapshots. Preserve licenses and
  record source versions/checksums in `third_party/SOURCES.json` on updates.
- Compile copied kernels with `MLIR_AIE_KERNEL_SOURCES=third_party/mlir-aie`.
- Put new RWKV-specific device kernels in `kernels/rwkv/`; do not silently change
  an upstream numerical contract or loosen tolerances to make a test pass.
- Build with CMake presets. Run hardware dispatches serially on the shared NPU.
- Keep build output in `build/`, detailed logs in `reports/runs/`, and commit only
  concise, dated validation summaries with hardware/software provenance.
- A compiled artifact is not a passing test: require successful C++ dispatch,
  numerical checks and guard checks. Report unsupported configurations separately.
- No model weights, build artifacts, caches, credentials or virtualenv in Git.

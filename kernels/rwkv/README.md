# RWKV AIE C++ kernels

| Source | Operation |
|---|---|
| `wkv7_fp32.cc` | Decode: one 64×64 `[key,value]` state transition and readout |
| `wkv7_prefill_fp32.cc` | Prefill: up to 16 transitions, padding preserves state |
| `gemv_fp32.cc` | FP32 projection tiles, all partial-sum arithmetic on NPU |
| `ops_fp32.cc` | Norm, mix, gates, activations, key normalization, residuals |

Build the complete runtime artifact bundle with
`.venv/bin/python tools/compile/rwkv7_full.py` from the repository root.
C++ calls XRT at runtime; Python only compiles and computes offline references.
See [inference.md](../../docs/inference.md) for layout, numerical methods,
constraints, graph/prefill separation and tests.

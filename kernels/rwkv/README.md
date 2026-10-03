# RWKV-7 production kernels

The supported pipeline uses BF16 matrix weights/inputs and FP32 accumulation,
recurrence, normalization outputs and residuals. Compile offline with
`.venv/bin/python tools/compile/rwkv7_optimized.py`; C++ alone dispatches the NPU.

| Source | Responsibility |
|---|---|
| `attention_projections_bf16.cc`, `rank_bf16.cc`, `gemv_bf16.cc` | Combined R/K/V and low-rank projections; pinned upstream BF16 MAC |
| `channel_mix_bf16.cc`, `ffn_pipeline_bf16.cc` | FFN key/ReLU²/value/residual and output projection helpers |
| `head_bf16.cc` | Full vocabulary projection |
| `norm_fp32.cc`, `norm_mix_fp32.cc`, `mix_fp32.cc`, `mix_pair_fp32.cc` | Upstream LayerNorm and shift/mix |
| `recurrence_stage_fp32.cc`, `stages_fp32.cc`, `wkv7_vector_fp32.cc` | Prepare/WKV/finish with FP32 persistent state |
| `value_fp32.cc` | First-value residual |
| `math_fp32.hpp` | Shared native exp/tanh/sigmoid and preserved double square-root helper |

No standalone FP32 GEMV, FP32 rank, scalar WKV experiment, or sequence-prefill
program remains. FP32 names on the retained recurrence and arithmetic helpers
are intentional. Upstream files and numerical tolerances remain unchanged.

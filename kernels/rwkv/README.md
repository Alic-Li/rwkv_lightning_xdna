# RWKV-7 production kernels

The supported pipeline uses BF16 matrix weights/inputs and FP32 accumulation,
recurrence, normalization outputs and residuals. Compile offline with
`.venv/bin/python tools/compile/rwkv7_optimized.py`; C++ alone dispatches the NPU.

| Source | Responsibility |
|---|---|
| `attention_projections_bf16.cc`, `rank_bf16.cc`, `gemv_bf16.cc` | Combined R/K/V and low-rank projections; pinned upstream BF16 MAC |
| `channel_mix_bf16.cc`, `ffn_pipeline_bf16.cc` | FFN key/ReLU²/value/residual and output projection helpers |
| `channel_mix_int8.cc`, `channel_mix_convert_bf16.cc`, `gemv_preconverted_bf16.hpp` | Experimental W8A16 FFN; producer-side BF16 activation rounding, on-core weight conversion and FP32 scale epilogue |
| `projection_residual_int8.cc` | Experimental W8A16 attention output; streamed tile conversion, FP32 accumulation, row-scale epilogue and residual |
| `ffn_common.hpp` | FFN helpers, including exact FP32 ReLU-square via vector integer significands; scalar handling preserves underflow/overflow |
| `head_bf16.cc` | Full vocabulary projection |
| `prefill_projection_bf16.cc` | Experimental batch-one/two BF16 FFN projections; reuse each streamed weight tile across tokens, token-major FP32 output |
| `prefill_channel_bf16.cc`, `prefill_mix_bf16.cc` | Experimental fused two-token FFN, with sequential shift carry, reused weight tiles, and one submission |
| `relu_squared_fp32.hpp` | Shared exact vector ReLU-square for production 2048-element and prefill 32-element blocks |
| `norm_fp32.cc`, `norm_mix_fp32.cc`, `mix_fp32.cc`, `mix_pair_fp32.cc` | Upstream LayerNorm and shift/mix |
| `recurrence_{prepare,update,finish}_fp32.cc`, `stages_fp32.cc`, `wkv7_vector_fp32.cc` | Prepare/WKV/finish with FP32 persistent state |
| `vector_exp_fp32.hpp` | 32-lane FP32 range reduction and exponential polynomial; scalar handling preserves small normal/subnormal results |
| `value_fp32.cc` | First-value residual |
| `math_fp32.hpp` | Shared native exp/tanh/sigmoid and preserved double square-root helper |

No standalone FP32 GEMV, FP32 rank, or scalar WKV experiment remains.
The experimental prefill projection and fused FFN have separate compilers and
guarded C++ tests; they are not yet connected to the model's prefill path.
FP32 names on the retained recurrence and arithmetic helpers
are intentional. Upstream files and numerical tolerances remain unchanged.

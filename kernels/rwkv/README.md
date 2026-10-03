# RWKV-7 device kernels

C++ dispatches all model arithmetic. Python compiles the explicit stage manifest
in `tools/compile/rwkv7_optimized.py`; no Python NPU dispatch.

| Group | Sources / responsibility |
|---|---|
| Normalization/mix | `norm_fp32`, `norm_mix_fp32`, `mix_pair_fp32` |
| BF16 projections | `attention_projections_bf16`, `rank_bf16`, `gemv_bf16`, `head_bf16` |
| BF16 FFN/output | `channel_mix_bf16`, `ffn_pipeline_bf16` |
| INT8 FFN | `channel_mix_int8`, `channel_mix_convert_int8`, `activation_int8.hpp`: fixed W8A8 block-256 streaming, signed INT8 multiplication, INT32 dot, FP32 scaled partial sums |
| Recurrence | `recurrence_{prepare,update,finish}_fp32`, `value_recurrence_fp32`, `stages_fp32`, `wkv7_vector_fp32` |
| Decode fusion | `decode_recurrence_projection_bf16`, `decode_recurrence_project_finish` |
| BF16 batch2 | `prefill_{attention,channel,mix,output}_bf16`, paired recurrence and recurrence/output fusion |
| BF16 chunk4 | `chunk4_channel_bf16`, `chunk4_channel_common.hpp` |
| Shared math | `ffn_common.hpp`, `relu_squared_fp32.hpp`, `vector_exp_fp32.hpp`, `math_fp32.hpp` |

Normalization, nonlinear outputs, residuals and recurrent state stay FP32.
INT8 weight packing is in `src/inference/quantization`; its W2 K-major layout is
fixed. W8A16, INT8 output projection and alternate integer reduction/dataflow
variants have been removed. Tests retain independent numerical oracles and guards.
Pinned upstream numerical contracts and tolerances are unchanged.

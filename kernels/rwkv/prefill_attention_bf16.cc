// SPDX-License-Identifier: Apache-2.0
#include "attention_projections_bf16.cc"
extern "C" void rwkv7_prefill_attention_rank(const float *x, const bfloat16 *w,
                                            float *out, int row, int col) {
  for (int token = 0; token < 2; ++token)
    rwkv7_attention_rank_tile(x + token * 2048, w, out + token * 128, row, col);
}
extern "C" void rwkv7_prefill_attention_activate(const float *raw, float *out, int p) {
  for (int token = 0; token < 2; ++token)
    rwkv7_attention_activate128(raw + token * 128, out + token * 128, p);
}
extern "C" void rwkv7_prefill_attention_rkv(const float *x, const bfloat16 *w,
                                           float *out, int col) {
  for (int token = 0; token < 2; ++token)
    rwkv7_attention_rkv_tile(x + token * 2048, w, out + token * 16, col);
}
extern "C" void rwkv7_prefill_attention_second(const float *x, const bfloat16 *w, float *out) {
  // First-stage gather is [core, token, 128]. Reassemble each 256-vector
  // locally, preserving the original GEMV reduction and BF16 rounding.
  alignas(64) float input[256];
  for (int token = 0; token < 2; ++token) {
    for (int core = 0; core < 2; ++core)
      for (int i = 0; i < 128; i += 32)
        aie::store_v(input + core * 128 + i,
                     aie::load_v<32>(x + core * 256 + token * 128 + i));
    rwkv7_gemv_tile(input, w, out + token * 16);
  }
}
extern "C" void rwkv7_prefill_attention_zero32(float *out) {
  aie::store_v(out, aie::zeros<float, 32>());
}
extern "C" void rwkv7_prefill_attention_zero256(float *out) {
  for (int i = 0; i < 256; i += 32) aie::store_v(out + i, aie::zeros<float, 32>());
}

// SPDX-License-Identifier: Apache-2.0
#include "rank_bf16.cc"
extern "C" void rwkv7_attention_zero128(float *out) {
  for (int i = 0; i < 128; ++i)
    out[i] = 0;
}
extern "C" void rwkv7_attention_rank_tile(const float *x, const bfloat16 *w,
                                          float *out, int row, int col) {
  rwkv7_gemv_tile(x + col * 256, w, out + row * 16);
}
extern "C" void rwkv7_attention_activate128(const float *raw, float *out,
                                            int projection) {
  rwkv7_rank_batch_activate64(raw, out, projection);
  rwkv7_rank_batch_activate64(raw + 64, out + 64, projection);
}
extern "C" void rwkv7_attention_rkv_tile(const float *x, const bfloat16 *w,
                                         float *out, int col) {
  rwkv7_gemv_tile(x + col * 256, w, out);
}

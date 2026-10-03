// SPDX-License-Identifier: Apache-2.0
#include "stages_fp32.cc"
#include "wkv7_vector_fp32.cc"
extern "C" void rwkv7_decode_recurrent(const float *state, const float *aux, float *out) {
  rwkv7_wkv_fp32(state, aux + 13 * 64, out);
}
extern "C" void rwkv7_decode_finish_projection(const float *aux, const float *y,
                                               float *out, int head) {
  alignas(32) float p[512], result[384];
  for (int j = 0; j < 64; ++j) {
    p[j] = y[j];
    p[64 + j] = aux[9 * 64 + j];
    p[128 + j] = aux[10 * 64 + j];
    p[192 + j] = aux[13 * 64 + j];
    p[256 + j] = aux[15 * 64 + j];
    p[320 + j] = aux[11 * 64 + j];
    p[384 + j] = aux[16 * 64 + j];
    p[448 + j] = aux[12 * 64 + j];
  }
  rwkv7_finish_head(p, result);
  for (int j = 0; j < 64; ++j) out[head * 64 + j] = result[192 + j];
}
extern "C" void rwkv7_decode_finish_projection256(const float *aux, const float *y,
                                                  float *out, int head) {
  rwkv7_decode_finish_projection(aux, y, out, head);
}

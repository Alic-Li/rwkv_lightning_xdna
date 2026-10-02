// SPDX-License-Identifier: Apache-2.0
#include "stages_fp32.cc"
#ifdef RWKV_EXACT_FP32
#include "wkv7_fp32.cc"
#else
#include "wkv7_vector_fp32.cc"
#endif

// One head of a shared 27-vector arena: prepare inputs [0,8), finish
// inputs [8,13), recurrent vectors [13,19), prepare outputs [19,21),
// finish outputs [21,27). Preserve every unchanged slot, including constants.
extern "C" void rwkv7_stage_prepare(const float *aux, float *updated) {
  alignas(32) float prepared[384];
  rwkv7_prepare_head(aux, prepared);
  for (int i = 0; i < 27 * 64; ++i)
    updated[i] = aux[i];
  for (int j = 0; j < 64; ++j) {
    updated[14 * 64 + j] = prepared[320 + j];
    updated[15 * 64 + j] = prepared[256 + j];
    updated[17 * 64 + j] = prepared[128 + j];
    updated[18 * 64 + j] = prepared[192 + j];
    updated[19 * 64 + j] = prepared[j];
    updated[20 * 64 + j] = prepared[64 + j];
  }
}
extern "C" void rwkv7_stage_recurrent(const float *state, const float *aux,
                                      float *out) {
  rwkv7_wkv_fp32(state, aux + 13 * 64, out);
  alignas(32) float y[64];
  for (int j = 0; j < 64; ++j)
    y[j] = out[4096 + j];
  float *updated = out + 4096;
  for (int i = 0; i < 27 * 64; ++i)
    updated[i] = aux[i];
  for (int j = 0; j < 64; ++j)
    updated[8 * 64 + j] = y[j];
}
extern "C" void rwkv7_stage_finish(const float *aux, float *updated) {
  alignas(32) float p[512], result[384];
  for (int j = 0; j < 64; ++j) {
    p[j] = aux[8 * 64 + j];
    p[64 + j] = aux[9 * 64 + j];
    p[128 + j] = aux[10 * 64 + j];
    p[192 + j] = aux[13 * 64 + j];
    p[256 + j] = aux[15 * 64 + j];
    p[320 + j] = aux[11 * 64 + j];
    p[384 + j] = aux[16 * 64 + j];
    p[448 + j] = aux[12 * 64 + j];
  }
  rwkv7_finish_head(p, result);
  for (int i = 0; i < 27 * 64; ++i)
    updated[i] = aux[i];
  for (int j = 0; j < 64; ++j)
    for (int v = 0; v < 4; ++v)
      updated[(21 + v) * 64 + j] = result[v * 64 + j];
}

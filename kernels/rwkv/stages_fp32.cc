// SPDX-License-Identifier: Apache-2.0
#include "math_fp32.hpp"
// Per-head fused preparation, [k,a,d,k_k,k_a,a0,w0,pad].
extern "C" void rwkv7_prepare_head(const float *p, float *out) {
  double ss = 0;
  for (int j = 0; j < 64; ++j) {
    float u = p[j] * p[192 + j];
    ss += (double)u * u;
  }
  float den = (float)root(ss);
  if (den < 1e-12f)
    den = 1e-12f;
  for (int j = 0; j < 64; ++j) {
    float kk = p[j] * p[192 + j] / den, a = sigmoid(p[64 + j] + p[320 + j]);
    out[j] = kk;
    out[64 + j] = a;
    out[128 + j] = -kk;
    out[192 + j] = kk * a;
    out[256 + j] = p[j] * (1 + (a - 1) * p[256 + j]);
    out[320 + j] =
        expf(-0.6065306597126334f * sigmoid(p[128 + j] + p[384 + j]));
  }
}
// [y,ln_weight,ln_bias,r,k,r_k,v,g] -> norm,residual,sum,gated.
extern "C" void rwkv7_finish_head(const float *p, float *out) {
  double mean = 0, var = 0;
  for (int j = 0; j < 64; ++j)
    mean += p[j];
  mean /= 64;
  for (int j = 0; j < 64; ++j) {
    double d = p[j] - mean;
    var += d * d;
  }
  double scale = 1.0 / root(var / 64 + 64e-5f);
  float dot = 0;
  for (int j = 0; j < 64; ++j)
    dot += p[192 + j] * p[256 + j] * p[320 + j];
  for (int j = 0; j < 64; ++j) {
    float norm = (float)((p[j] - mean) * scale) * p[64 + j] + p[128 + j];
    float res = dot * p[384 + j];
    out[j] = norm;
    out[64 + j] = res;
    out[128 + j] = norm + res;
    out[192 + j] = (norm + res) * p[448 + j];
    out[256 + j] = 0;
    out[320 + j] = 0;
  }
}
// Four heads per worker. Layouts match strided DMA directly; no many-input
// memory-tile joins or host packing between stages.
extern "C" void rwkv7_prepare_dma(const float *a, const float *b, float *tmp,
                                  float *rec) {
  float p[512], out[384];
  for (int h = 0; h < 4; ++h) {
    for (int j = 0; j < 4; ++j)
      for (int k = 0; k < 64; ++k) {
        p[j * 64 + k] = a[j * 256 + h * 64 + k];
        p[256 + j * 64 + k] = b[j * 256 + h * 64 + k];
      }
    rwkv7_prepare_head(p, out);
    for (int k = 0; k < 64; ++k) {
      int i = h * 64 + k;
      tmp[i] = out[k];
      tmp[256 + i] = out[64 + k];
      rec[i] = out[320 + k];
      rec[256 + i] = out[256 + k];
      rec[512 + i] = out[128 + k];
      rec[768 + i] = out[192 + k];
    }
  }
}
extern "C" void rwkv7_finish_dma(const float *a, const float *rec,
                                 float *result) {
  float p[512], out[384];
  for (int h = 0; h < 4; ++h) {
    for (int k = 0; k < 64; ++k) {
      int i = h * 64 + k;
      p[k] = a[i];
      p[64 + k] = a[256 + i];
      p[128 + k] = a[512 + i];
      p[192 + k] = rec[i];
      p[256 + k] = rec[512 + i];
      p[320 + k] = a[768 + i];
      p[384 + k] = rec[768 + i];
      p[448 + k] = a[1024 + i];
    }
    rwkv7_finish_head(p, out);
    for (int j = 0; j < 4; ++j)
      for (int k = 0; k < 64; ++k)
        result[j * 256 + h * 64 + k] = out[j * 64 + k];
  }
}

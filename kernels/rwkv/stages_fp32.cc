// SPDX-License-Identifier: Apache-2.0
#include "vector_exp_fp32.hpp"
// Per-head fused preparation, [k,a,d,k_k,k_a,a0,w0,pad].
static void rwkv7_prepare_head(const float *p, float *out) {
  double ss = 0;
  for (int j = 0; j < 64; ++j) {
    float u = p[j] * p[192 + j];
    ss += (double)u * u;
  }
  float den = (float)root(ss);
  if (den < 1e-12f)
    den = 1e-12f;
  alignas(32) float gate[64], decay[64];
  sigmoid64(p + 64, p + 320, gate);
  sigmoid64(p + 128, p + 384, decay);
  for (int j = 0; j < 64; ++j)
    decay[j] = -0.6065306597126334f * decay[j];
  exp_negative32(decay, out + 320);
  exp_negative32(decay + 32, out + 352);
  for (int j = 0; j < 64; ++j) {
    float kk = p[j] * p[192 + j] / den, a = gate[j];
    out[j] = kk;
    out[64 + j] = a;
    out[128 + j] = -kk;
    out[192 + j] = kk * a;
    out[256 + j] = p[j] * (1 + (a - 1) * p[256 + j]);
  }
}
// [y,ln_weight,ln_bias,r,k,r_k,v,g] -> norm,residual,sum,gated.
static void rwkv7_finish_head(const float *p, float *out) {
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

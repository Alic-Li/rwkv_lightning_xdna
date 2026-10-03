// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "math_fp32.hpp"

// Same range reduction and seventh-order polynomial as expf, batched across
// 32 lanes. Callers supply nonpositive finite values. The scalar path below
// preserves the original treatment of small normal and subnormal results.
static void exp_negative32(const float *input, float *output) {
  alignas(32) float clipped[32], exponent[32], power[32];
  for (int i = 0; i < 32; ++i)
    clipped[i] = input[i] < -69.f ? -69.f : input[i];
  auto x = aie::load_v<32>(clipped);
  auto scaled = aie::mul(x, 1.4426950408889634f).to_vector<float>();
  for (int i = 0; i < 32; ++i) {
    float s = scaled[i];
    int k = (int)(s + (s >= 0 ? 0.5f : -0.5f));
    exponent[i] = aie::to_float<float>(k);
    union { uint32_t bits; float value; } p;
    p.bits = (uint32_t)(k + 127) << 23;
    power[i] = p.value;
  }
  auto fk = aie::load_v<32>(exponent);
  auto r = aie::sub(aie::sub(x, aie::mul(fk, 0.693145751953125f).to_vector<float>()),
                    aie::mul(fk, 1.428606765330187e-6f).to_vector<float>());
  auto p = aie::broadcast<float, 32>(1.f / 5040.f);
  const float coefficients[] = {1.f / 720.f, 1.f / 120.f, 1.f / 24.f,
                                1.f / 6.f, 0.5f, 1.f, 1.f};
  for (float c : coefficients)
    p = aie::add(aie::broadcast<float, 32>(c), aie::mul(r, p).to_vector<float>());
  aie::store_v(output, aie::mul(p, aie::load_v<32>(power)).to_vector<float>());
  for (int i = 0; i < 32; ++i)
    if (input[i] < -69.f)
      output[i] = expf(input[i]);
}

static void sigmoid64(const float *x, const float *bias, float *output) {
  alignas(32) float sum[32], negative[32], e[32], numerator[32], reciprocal[32];
  for (int offset = 0; offset < 64; offset += 32) {
    for (int i = 0; i < 32; ++i) {
      sum[i] = x[offset + i] + bias[offset + i];
      negative[i] = sum[i] >= 0 ? -sum[i] : sum[i];
    }
    exp_negative32(negative, e);
    for (int i = 0; i < 32; ++i) {
      numerator[i] = sum[i] >= 0 ? 1.f : e[i];
      reciprocal[i] = ::aie::inv(1.f + e[i]);
    }
    // fp32_div already uses vector multiplication through one lane. Batch
    // that same operation without changing the native scalar reciprocal.
    aie::store_v(output + offset,
        aie::mul(aie::load_v<32>(numerator), aie::load_v<32>(reciprocal)).to_vector<float>());
  }
}

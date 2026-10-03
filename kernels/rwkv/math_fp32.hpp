// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>
#include "../../third_party/mlir-aie/aie_kernels/common/scalar_f32.h"
static float fp32_div(float x, float y) { return scalar_mul(x, ::aie::inv(y)); }
// FP32 range reduction; seventh-order polynomial on [-ln(2)/2, ln(2)/2].
// Preserve subnormal exp outputs by splitting the power of two.
static float expf(float x) {
  if (x < -104.f)
    return 0.f;
  if (x > 88.f)
    x = 88.f;
  float scaled = scalar_mul(x, 1.4426950408889634f);
  int k = (int)(scaled + (scaled >= 0 ? 0.5f : -0.5f));
  float fk = ::aie::to_float<float>(k);
  float r = (x - scalar_mul(fk, 0.693145751953125f)) -
            scalar_mul(fk, 1.428606765330187e-6f);
  float p = 1.f / 5040.f;
  p = 1.f / 720.f + scalar_mul(r, p);
  p = 1.f / 120.f + scalar_mul(r, p);
  p = 1.f / 24.f + scalar_mul(r, p);
  p = 1.f / 6.f + scalar_mul(r, p);
  p = 0.5f + scalar_mul(r, p);
  p = 1.f + scalar_mul(r, p);
  p = 1.f + scalar_mul(r, p);
  union {
    uint32_t bits;
    float value;
  } power;
  if (k < -126) {
    power.bits = (uint32_t)(k + 126 + 127) << 23;
    return (p * power.value) * 0x1p-126f;
  }
  power.bits = (uint32_t)(k + 127) << 23;
  if (k < -100)
    return p * power.value;
  return scalar_mul(p, power.value);
}
static float tanhf(float x) {
  float ax = x < 0 ? -x : x;
  // Stable around zero: exp(-2*x) subtraction otherwise loses low bits.
  if (ax < 0.01f) {
    float x2 = scalar_mul(x, x);
    return scalar_mul(x, 1.f - scalar_mul(x2, 1.f / 3.f) +
                             scalar_mul(scalar_mul(x2, x2), 2.f / 15.f));
  }
  float e = expf(scalar_mul(-2.f, ax));
  float t = fp32_div(1.f - e, 1.f + e);
  return x < 0 ? -t : t;
}

static double root(double x) {
  if (x <= 0)
    return 0;
  union {
    uint64_t bits;
    double value;
  } u;
  u.value = x;
  u.bits = (u.bits >> 1) + 0x1ff8000000000000ULL;
  double y = u.value;
  for (int i = 0; i < 7; ++i)
    y = 0.5 * (y + x / y);
  return y;
}
// ABI: [opcode, count, group, epsilon, reserved x12, x[2048], y, z, w].
static float sigmoid(float x) {
  if (x >= 0)
    return fp32_div(1.f, 1.f + expf(-x));
  float e = expf(x);
  return fp32_div(e, 1.f + e);

}

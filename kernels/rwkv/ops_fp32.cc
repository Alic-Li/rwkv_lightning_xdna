// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#ifdef RWKV_FAST_EXP
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
#else
// Range-reduced exp evaluated in double; final results are rounded to float.
// Degree 12 on [-ln(2)/2, ln(2)/2], no BF16 approximation.
static double exponential(double x) {
  if (x < -100.0)
    return 0;
  if (x > 88.0)
    x = 88.0;
  double scaled = x * 1.4426950408889634074;
  int k = (int)(scaled + (scaled >= 0 ? 0.5 : -0.5));
  double r = x - k * 0.69314718055994530942;
  double term = 1, sum = 1;
  for (int i = 1; i <= 12; ++i) {
    term *= r / i;
    sum += term;
  }
  union {
    uint64_t bits;
    double value;
  } power;
  power.bits = (uint64_t)(k + 1023) << 52;
  return sum * power.value;
}
static float expf(float x) { return (float)exponential(x); }
static float tanhf(float x) {
  double ax = x < 0 ? -(double)x : x;
  double e = exponential(-2 * ax);
  float t = (float)((1 - e) / (1 + e));
  return x < 0 ? -t : t;
}
#endif
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
#ifdef RWKV_FAST_EXP
  if (x >= 0)
    return fp32_div(1.f, 1.f + expf(-x));
  float e = expf(x);
  return fp32_div(e, 1.f + e);
#else
  if (x >= 0)
    return 1.f / (1.f + expf(-x));
  float e = expf(x);
  return e / (1.f + e);
#endif
}
extern "C" void rwkv7_ops_fp32(const float *p, float *out) {
  const int op = (int)p[0], count = (int)p[1], group = (int)p[2];
  const float *x = p + 16, *y = x + 2048, *z = y + 2048, *w = z + 2048;
  for (int i = 0; i < 2048; ++i)
    out[i] = 0;
  if (op == 9 || op == 10 || op == 11) {
    for (int start = 0; start < count; start += group) {
      if (op == 9) {
        double ss = 0;
        for (int j = 0; j < group; ++j) {
          float u = x[start + j] * y[start + j];
          ss += (double)u * u;
        }
        float den = (float)root(ss);
        if (den < 1e-12f)
          den = 1e-12f;
        for (int j = 0; j < group; ++j)
          out[start + j] = (x[start + j] * y[start + j]) / den;
      } else if (op == 10) {
        float dot = 0;
        for (int j = 0; j < group; ++j)
          dot += x[start + j] * y[start + j] * z[start + j];
        for (int j = 0; j < group; ++j)
          out[start + j] = dot * w[start + j];
      } else {
        double mean = 0, variance = 0;
        for (int j = 0; j < group; ++j)
          mean += x[start + j];
        mean /= group;
        for (int j = 0; j < group; ++j) {
          double d = x[start + j] - mean;
          variance += d * d;
        }
        double scale = 1.0 / root(variance / group + p[3]);
        for (int j = 0; j < group; ++j)
          out[start + j] =
              (float)((x[start + j] - mean) * scale) * y[start + j] +
              z[start + j];
      }
    }
    return;
  }
  for (int i = 0; i < count; ++i) {
    switch (op) {
    case 0:
      out[i] = x[i] + y[i];
      break;
    case 1:
      out[i] = x[i] + (y[i] - x[i]) * z[i];
      break;
    case 2:
      out[i] = tanhf(x[i]);
      break;
    case 3:
      out[i] = sigmoid(x[i] + y[i]);
      break;
    case 4:
      out[i] = expf(-0.6065306597126334f * sigmoid(x[i] + y[i]));
      break;
    case 5:
      out[i] = x[i] * y[i];
      break;
    case 6:
      out[i] = x[i] + (y[i] - x[i]) * sigmoid(z[i] + w[i]);
      break;
    case 7: {
      float a = x[i] > 0 ? x[i] : 0;
      out[i] = a * a;
      break;
    }
    case 8:
      out[i] = x[i] * (1 + (y[i] - 1) * z[i]);
      break;
    case 12:
      out[i] = -x[i];
      break;
    }
  }
}

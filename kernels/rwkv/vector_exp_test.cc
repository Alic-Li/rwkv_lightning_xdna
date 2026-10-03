// SPDX-License-Identifier: Apache-2.0
#include "vector_exp_fp32.hpp"
extern "C" void rwkv7_vector_exp_test(const float *input, float *out) {
  alignas(32) float negative[64], zero[64] = {}, candidate[64];
  for (int i = 0; i < 64; ++i) {
    negative[i] = input[i] >= 0 ? -input[i] : input[i];
    out[i] = expf(negative[i]);
    out[128 + i] = sigmoid(input[i]);
  }
  exp_negative32(negative, out + 64);
  exp_negative32(negative + 32, out + 96);
  sigmoid64(input, zero, candidate);
  for (int i = 0; i < 64; ++i) out[192 + i] = candidate[i];
}

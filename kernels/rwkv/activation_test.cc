// SPDX-License-Identifier: Apache-2.0
#include "ffn_common.hpp"
extern "C" void rwkv7_activation_test(const float *input, float *out) {
  for (int i = 0; i < 2048; ++i) {
    float x = input[i] > 0 ? input[i] : 0;
    out[i] = x * x;
  }
  rwkv7_ffn_activate2048(input, out + 2048);
  for (int i = 0; i < 2048; i += 32)
    relu_squared_fp32<32>(input + i, out + 4096 + i);
}

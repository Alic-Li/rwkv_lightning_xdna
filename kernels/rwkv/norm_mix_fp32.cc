// SPDX-License-Identifier: Apache-2.0
#include "norm_fp32.cc"
extern "C" void rwkv7_norm_pair(const float *input, float *out) {
  rwkv7_layer_norm(input, out);
  for (int i = 0; i < 2048; i += 32)
    aie::store_v(out + 2048 + i, aie::load_v<32>(input + 6144 + i));
}

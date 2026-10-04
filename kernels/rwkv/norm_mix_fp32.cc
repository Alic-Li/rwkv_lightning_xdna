// SPDX-License-Identifier: Apache-2.0
#include "model_shape.hpp"
#include "norm_fp32.cc"
extern "C" void rwkv7_norm_pair(const float *input, float *out) {
  rwkv7_layer_norm(input, out);
  for (int i = 0; i < RWKV_CHANNELS; i += 32)
    aie::store_v(out + RWKV_CHANNELS + i, aie::load_v<32>(input + (3 * RWKV_CHANNELS) + i));
}

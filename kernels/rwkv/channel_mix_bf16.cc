// SPDX-License-Identifier: Apache-2.0
#include "ffn_pipeline_bf16.cc"
extern "C" void rwkv7_channel_zero512(float *out) {
  for (int i = 0; i < 512; i += 32)
    aie::store_v(out + i, aie::zeros<float, 32>());
}

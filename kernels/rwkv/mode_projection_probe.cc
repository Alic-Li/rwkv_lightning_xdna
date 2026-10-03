// SPDX-License-Identifier: Apache-2.0
#include "ffn_pipeline_bf16.cc"
#include "norm_fp32.cc"
extern "C" void rwkv7_mode_norm_slice(const float *packed, float *out,
                                      int lane) {
  alignas(64) float normalized[2048];
  rwkv7_layer_norm(packed, normalized);
  for (int i = 0; i < 256; i += 16)
    aie::store_v(out + i, aie::load_v<16>(normalized + lane * 256 + i));
}

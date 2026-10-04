// SPDX-License-Identifier: Apache-2.0
#include "model_shape.hpp"
#include "mix_fp32.cc"
extern "C" void rwkv7_mix_pair(const float *pair, const float *coeff,
                               float *out) {
  rwkv7_mix(pair, pair + RWKV_CHANNELS, coeff, out);
}

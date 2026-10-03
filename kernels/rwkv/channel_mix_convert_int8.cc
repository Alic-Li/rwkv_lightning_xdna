// SPDX-License-Identifier: Apache-2.0
#include "activation_int8.hpp"
extern "C" void rwkv7_channel_convert2048(const float *x, uint8_t *out) {
  convert<2048>(x, out);
}
extern "C" void rwkv7_channel_convert8192(const float *x, uint8_t *out) {
  convert<8192>(x, out);
}

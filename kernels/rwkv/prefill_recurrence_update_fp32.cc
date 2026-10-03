// SPDX-License-Identifier: Apache-2.0
#include "wkv7_vector_fp32.cc"
extern "C" void rwkv7_pair_recurrent(const float *state, const float *aux, float *out) {
  rwkv7_wkv_fp32(state, aux + 13 * 64, out);
  alignas(64) float first_y[64];
  for (int j = 0; j < 64; j += 32)
    aie::store_v(first_y + j, aie::load_v<32>(out + 4096 + j));
  // Each 32-column block reads all old rows before overwriting them; the
  // second update can reuse the first update's state storage without changing
  // the key accumulation order. Output: final state, token-0 y, token-1 y.
  rwkv7_wkv_fp32(out, aux + 1728 + 13 * 64, out);
  for (int j = 0; j < 64; j += 32) {
    aie::store_v(out + 4160 + j, aie::load_v<32>(out + 4096 + j));
    aie::store_v(out + 4096 + j, aie::load_v<32>(first_y + j));
  }
}

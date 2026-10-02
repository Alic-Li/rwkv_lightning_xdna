// SPDX-License-Identifier: Apache-2.0
#include <aie_api/aie.hpp>
extern "C" void rwkv7_stream_copy(const float *input, float *output,
                                  int count) {
  for (int i = 0; i < count; i += 32)
    aie::store_v(output + i, aie::load_v<32>(input + i));
}
extern "C" void rwkv7_packet_copy(const float *input, float *output,
                                  int offset) {
  rwkv7_stream_copy(input + 2048 * offset, output, 2048);
}

// SPDX-License-Identifier: Apache-2.0
#include <aie_api/aie.hpp>
// Out-of-line diagnostic calls keep markers behind the caller's FIFO acquire.
extern "C" void rwkv7_stream_trace_start(const float *) {
  event0();
}
extern "C" void rwkv7_stream_trace_end(const float *) {
  event1();
}

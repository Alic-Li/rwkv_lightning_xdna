// SPDX-License-Identifier: Apache-2.0
#include "model_shape.hpp"
// Reuse the pinned upstream two-pass FP32 implementation without BF16 cast.
#include "../../third_party/mlir-aie/aie_kernels/transformer/layer_norm_f32.cc"
extern "C" void rwkv7_layer_norm(const float *p, float *out) {
  layer_norm_f32_impl<float, float, 16, true>(p, out, p + RWKV_CHANNELS, p + (2 * RWKV_CHANNELS), RWKV_CHANNELS);
}

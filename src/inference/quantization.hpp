// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "rwkv/inference/model.hpp"
#include <cstdint>

namespace rwkv::inference::quantization {
// CUDA RWKV W8A16 contract: symmetric per-output rows, codes [-127,127],
// scales rounded using the checkpoint I/O FP16 conversion before quantizing.
// FP16 scales are expanded exactly to FP32 for device epilogues.
struct Rows {
  size_t outputs = 0, inputs = 0;
  std::vector<int8_t> codes;
  Vector scales;
};
Rows quantize(const Tensor &, bool transposed = false);
// [row/16,K/256] tiles: 4096 signed bytes followed by 16 FP32 scales.
// Repeated row scales keep each streamed tile self-contained and 64B aligned.
constexpr size_t tile_bytes = 4160;
std::vector<uint8_t> pack(const Rows &);
std::vector<uint8_t> channel_mix(const Tensor &key, const Tensor &value);
} // namespace rwkv::inference::quantization

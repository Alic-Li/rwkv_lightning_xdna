// SPDX-License-Identifier: Apache-2.0
#include "quantization.hpp"
#include "rwkv/io/pth_tensor.hpp"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace rwkv::inference::quantization {
Rows quantize(const Tensor &tensor, bool transposed) {
  if (tensor.shape.size() != 2 || !tensor.shape[0] || !tensor.shape[1] ||
      tensor.shape[0] > std::numeric_limits<size_t>::max() / tensor.shape[1] ||
      tensor.data.size() != tensor.shape[0] * tensor.shape[1])
    throw std::invalid_argument("Invalid W8A16 matrix shape");
  if (std::fegetround() != FE_TONEAREST)
    throw std::invalid_argument("W8A16 packing requires FE_TONEAREST");
  Rows result;
  result.outputs = tensor.shape[transposed ? 1 : 0];
  result.inputs = tensor.shape[transposed ? 0 : 1];
  result.codes.resize(tensor.data.size());
  result.scales.resize(result.outputs);
  auto element = [&](size_t row, size_t col) {
    return tensor.data[transposed ? col * result.outputs + row
                                  : row * result.inputs + col];
  };
  for (size_t row = 0; row < result.outputs; ++row) {
    float max_abs = 0;
    for (size_t col = 0; col < result.inputs; ++col) {
      float value = element(row, col);
      if (!std::isfinite(value))
        throw std::invalid_argument("Nonfinite W8A16 weight");
      max_abs = std::max(max_abs, std::abs(value));
    }
    float unrounded = std::max(max_abs / 127.f, std::numeric_limits<float>::min());
    float scale = std::max(llm_infer::f16_bits_to_float(
        llm_infer::float_to_f16_bits(unrounded)), 0x1p-24f);
    if (!std::isfinite(scale))
      throw std::invalid_argument("W8A16 FP16 scale overflow");
    result.scales[row] = scale;
    for (size_t col = 0; col < result.inputs; ++col) {
      float code = std::nearbyint(element(row, col) / scale);
      result.codes[row * result.inputs + col] =
          static_cast<int8_t>(std::clamp(code, -127.f, 127.f));
    }
  }
  return result;
}
std::vector<uint8_t> pack(const Rows &rows) {
  if (!rows.outputs || !rows.inputs || rows.outputs % 16 || rows.inputs % 256 ||
      rows.outputs > std::numeric_limits<size_t>::max() / rows.inputs ||
      rows.codes.size() != rows.outputs * rows.inputs ||
      rows.scales.size() != rows.outputs)
    throw std::invalid_argument("Invalid tiled W8A16 shape");
  size_t tiles = rows.outputs * rows.inputs / 4096;
  if (tiles > std::numeric_limits<size_t>::max() / tile_bytes)
    throw std::invalid_argument("W8A16 packed size overflow");
  for (float scale : rows.scales)
    if (!std::isfinite(scale) || scale <= 0)
      throw std::invalid_argument("Invalid W8A16 scale");
  for (int8_t code : rows.codes)
    if (code == -128)
      throw std::invalid_argument("W8A16 code outside symmetric range");
  std::vector<uint8_t> packed(tiles * tile_bytes);
  for (size_t row = 0; row < rows.outputs; row += 16)
    for (size_t col = 0; col < rows.inputs; col += 256) {
      size_t tile = (row / 16) * (rows.inputs / 256) + col / 256;
      auto *dst = packed.data() + tile * tile_bytes;
      for (size_t r = 0; r < 16; ++r)
        std::memcpy(dst + r * 256, rows.codes.data() + (row + r) * rows.inputs + col, 256);
      std::memcpy(dst + 4096, rows.scales.data() + row, 16 * sizeof(float));
    }
  return packed;
}
std::vector<uint8_t> channel_mix(const Tensor &key, const Tensor &value) {
  if (key.shape != std::vector<size_t>{8192, 2048} ||
      value.shape != std::vector<size_t>{2048, 8192})
    throw std::invalid_argument("Invalid W8A16 ChannelMix shape");
  auto packed = pack(quantize(key));
  auto second = pack(quantize(value));
  packed.insert(packed.end(), second.begin(), second.end());
  return packed;
}
} // namespace rwkv::inference::quantization

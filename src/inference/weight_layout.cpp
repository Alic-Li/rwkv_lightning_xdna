// SPDX-License-Identifier: Apache-2.0
#include "weight_layout.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
namespace rwkv::inference::weight_layout {
namespace {
void append_projection(Vector &packed, const Tensor &logical, bool transposed,
                       size_t first_output, size_t rows, size_t reduction) {
  if (logical.shape.size() != 2 ||
      logical.data.size() != logical.shape[0] * logical.shape[1])
    throw std::invalid_argument("Invalid logical projection shape");
  const size_t inputs = logical.shape[transposed ? 0 : 1],
               outputs = logical.shape[transposed ? 1 : 0];
  if (!rows || rows % row_tile || reduction % reduction_tile ||
      reduction < inputs || first_output >= outputs)
    throw std::invalid_argument("Invalid physical projection shape");
  const size_t offset = packed.size();
  packed.resize(offset + rows * reduction, 0);
  for (size_t row = 0; row < std::min(rows, outputs - first_output); ++row)
    for (size_t col = 0; col < inputs; ++col) {
      size_t tile = (row / 16) * (reduction / 256) + col / 256;
      packed[offset + tile * 4096 + (row % 16) * 256 + col % 256] =
          logical.data[transposed ? col * outputs + first_output + row
                                  : (first_output + row) * inputs + col];
    }
}
} // namespace
Vector projection(const Tensor &logical, bool transposed, size_t first_output,
                  size_t rows, size_t reduction) {
  Vector packed;
  append_projection(packed, logical, transposed, first_output, rows, reduction);
  return packed;
}
Vector channel_mix(const Tensor &key, const Tensor &value) {
  const size_t c = key.shape.at(1), h = key.shape.at(0);
  if (key.shape != std::vector<size_t>{h, c} ||
      value.shape != std::vector<size_t>{c, h})
    throw std::invalid_argument("Invalid ChannelMix weights");
  Vector packed;
  packed.reserve(2 * c * h);
  append_projection(packed, key, false, 0, h, c);
  append_projection(packed, value, false, 0, c, h);
  return packed;
}
Vector rkv(const std::array<const Tensor *, 3> &logical) {
  if (!logical[0] || logical[0]->shape.size() != 2)
    throw std::invalid_argument("Invalid RKV weight");
  const size_t c = logical[0]->shape[0];
  Vector packed(3 * c * c);
  for (size_t projection = 0; projection < 3; ++projection) {
    const auto *matrix = logical[projection];
    if (!matrix || matrix->shape != std::vector<size_t>{c, c} ||
        matrix->data.size() != c * c)
      throw std::invalid_argument("Invalid RKV weight");
    for (size_t row = 0; row < c; ++row)
      for (size_t col = 0; col < c; ++col) {
        const size_t tile =
            ((row / (c / 8) * 3 + projection) * (c / 128) + (row % (c / 8)) / 16) * (c / 256) +
            col / 256;
        packed[tile * 4096 + (row % 16) * 256 + col % 256] =
            matrix->data[row * c + col];
      }
  }
  return packed;
}
Vector rank_batch(const std::vector<RankBranch> &branches, size_t first_workers,
                  size_t second_workers) {
  if (branches.empty() || !branches[0].input || branches[0].input->shape.size() != 2)
    throw std::invalid_argument("Invalid rank weights");
  const size_t c = branches[0].input->shape[0];
  if (branches.empty() || !first_workers || !second_workers ||
      256 % first_workers || c % second_workers ||
      (256 / first_workers) % 16 || (c / second_workers) % 16)
    throw std::invalid_argument("Invalid rank worker partition");
  const size_t count = branches.size(), first_stripe = 256 / first_workers,
               second_stripe = c / second_workers;
  Vector packed(count * 2 * 256 * c, 0);
  for (size_t branch = 0; branch < count; ++branch) {
    const auto *first = branches[branch].input,
               *second = branches[branch].output;
    if (!first || !second || first->shape.size() != 2 ||
        first->shape[0] != c || !first->shape[1] || first->shape[1] > 256 ||
        second->shape != std::vector<size_t>{first->shape[1], c} ||
        first->data.size() != c * first->shape[1] ||
        second->data.size() != first->data.size())
      throw std::invalid_argument("Invalid rank weights");
    const size_t hidden = first->shape[1];
    for (size_t row = 0; row < hidden; ++row)
      for (size_t col = 0; col < c; ++col) {
        const size_t tile =
            ((row / first_stripe * count + branch) * (first_stripe / 16) +
             (row % first_stripe) / 16) *
                (c / 256) +
            col / 256;
        packed[tile * 4096 + (row % 16) * 256 + col % 256] =
            first->data[col * hidden + row];
      }
    for (size_t row = 0; row < c; ++row)
      for (size_t col = 0; col < hidden; ++col) {
        const size_t tile =
            (row / second_stripe * count + branch) * (second_stripe / 16) +
            (row % second_stripe) / 16;
        packed[count * 256 * c + tile * 4096 + (row % 16) * 256 + col] =
            second->data[col * c + row];
      }
  }
  return packed;
}
std::vector<uint16_t> to_bf16(const Vector &packed) {
  std::vector<uint16_t> result(packed.size());
  for (size_t i = 0; i < packed.size(); ++i) {
    uint32_t bits;
    std::memcpy(&bits, &packed[i], 4);
    result[i] =
        static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
  }
  return result;
}
} // namespace rwkv::inference::weight_layout

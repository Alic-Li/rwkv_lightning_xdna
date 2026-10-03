// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "rwkv/inference/model.hpp"
#include <array>
#include <cstdint>

namespace rwkv::inference::weight_layout {
// Logical weights are float32 checkpoint tensors: [output,input], or
// [input,output] for transposed low-rank projections. Packing never mutates
// them. Standard physical order: [output/16,input/256,16,256]. Missing
// rows/columns are zero. XRT allocates aligned root BOs; every tile is 4096
// elements (8192 bytes BF16), satisfying the device's
// 64-byte alignment.
constexpr size_t row_tile = 16, reduction_tile = 256, tile_elements = 4096;
Vector projection(const Tensor &logical, bool transposed, size_t first_output,
                  size_t physical_outputs, size_t physical_inputs);
// Concatenated standard key[8192,2048], value[2048,8192] tiles.
// Experimental value layout: [worker=4,K/256,local_row/16,16,256].
Vector channel_mix(const Tensor &key, const Tensor &value, bool value_k_major = false);
// [worker=8,projection=3,local_row/16,K/256,16,256], stripe=256 rows.
Vector rkv(const std::array<const Tensor *, 3> &logical);
struct RankBranch {
  const Tensor *input;
  const Tensor *output;
};
// Concatenated first/second matrices. First physical shape:
// [first_workers,branches,256/first_workers/16,8,16,256]; second:
// [second_workers,branches,2048/second_workers/16,1,16,256].
// Logical hidden ranks <=256 are padded with zeros in both matrices.
Vector rank_batch(const std::vector<RankBranch> &branches, size_t first_workers,
                  size_t second_workers);
// Round-to-nearest-even storage conversion; FP32 accumulation/state stay FP32.
std::vector<uint16_t> to_bf16(const Vector &packed);
} // namespace rwkv::inference::weight_layout

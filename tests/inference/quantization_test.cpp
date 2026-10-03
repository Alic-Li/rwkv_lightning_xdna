// SPDX-License-Identifier: Apache-2.0
#include "quantization.hpp"
#include <cfenv>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace rwkv::inference;
namespace q = rwkv::inference::quantization;
static void check(bool ok) {
  if (!ok)
    throw std::runtime_error("INT8 quantization contract mismatch");
}
template <class F> static void rejects(F fn) {
  try {
    fn();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("Accepted invalid INT8 matrix");
}
int main() {
  try {
    Tensor t{{16, 256}, Vector(16 * 256, 0)};
    for (size_t r = 1; r < 16; ++r) {
      t.data[r * 256] = 127;
      t.data[r * 256 + 1] = .5f;
      t.data[r * 256 + 2] = 1.5f;
      t.data[r * 256 + 3] = 2.5f;
      t.data[r * 256 + 4] = -1.5f;
      t.data[r * 256 + 5] = -2.5f;
      t.data[r * 256 + 6] = -127;
    }
    auto rows = q::quantize(t);
    check(rows.scales[0] == 0x1p-24f);
    for (size_t r = 1; r < 16; ++r) {
      check(rows.scales[r] == 1);
      const int expected[] = {127, 0, 2, 2, -2, -2, -127};
      for (size_t c = 0; c < 7; ++c)
        check(rows.codes[r * 256 + c] == expected[c]);
    }
    auto packed = q::pack(rows);
    check(packed.size() == 4160);
    for (size_t i = 0; i < rows.codes.size(); ++i)
      check(packed[i] == static_cast<uint8_t>(rows.codes[i]));
    float scales[16];
    std::memcpy(scales, packed.data() + 4096, sizeof(scales));
    for (size_t i = 0; i < 16; ++i)
      check(scales[i] == rows.scales[i]);
    Tensor transpose{{256, 16}, Vector(t.data.size())};
    for (size_t r = 0; r < 16; ++r)
      for (size_t c = 0; c < 256; ++c)
        transpose.data[c * 16 + r] = t.data[r * 256 + c];
    check(q::pack(q::quantize(transpose, true)) == packed);
    Tensor tiled{{32, 512}, Vector(32 * 512)};
    for (size_t r = 0; r < 32; ++r)
      for (size_t c = 0; c < 512; ++c)
        tiled.data[r * 512 + c] =
            float(int((r * 13 + c) % 255) - 127) * (r + 1);
    auto tiled_rows = q::quantize(tiled);
    auto tiled_bytes = q::pack(tiled_rows);
    check(tiled_bytes.size() == 4 * q::tile_bytes);
    for (size_t r = 0; r < 32; ++r)
      for (size_t c = 0; c < 512; ++c) {
        size_t tile = r / 16 * 2 + c / 256;
        size_t offset = tile * q::tile_bytes + r % 16 * 256 + c % 256;
        check(tiled_bytes[offset] == uint8_t(tiled_rows.codes[r * 512 + c]));
        float scale;
        std::memcpy(
            &scale,
            tiled_bytes.data() + tile * q::tile_bytes + 4096 + r % 16 * 4, 4);
        check(scale == tiled_rows.scales[r]);
      }
    // Match CUDA's I/O helper at a scale halfway between FP16 values: its
    // mantissa conversion rounds this tie up, unlike IEEE nearest-even.
    t.data[256] = 127.f * (1.f + 0x1p-11f);
    check(q::quantize(t).scales[1] == 1.f + 0x1p-10f);
    t.data[0] = 0x1p-30f;
    auto tiny = q::quantize(t);
    check(tiny.scales[0] == 0x1p-24f && tiny.codes[0] == 0);
    t.data[0] = std::numeric_limits<float>::infinity();
    rejects([&] { q::quantize(t); });
    t.data[0] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { q::quantize(t); });
    t.data[0] = std::numeric_limits<float>::max();
    rejects([&] { q::quantize(t); });
    rejects([&] { q::quantize(Tensor{{0, 256}, {}}); });
    rejects([&] { q::quantize(Tensor{{2, 2}, {1}}); });
    int rounding = std::fegetround();
    std::fesetround(FE_UPWARD);
    bool rejected_rounding = false;
    try {
      q::quantize(tiled);
    } catch (const std::invalid_argument &) {
      rejected_rounding = true;
    }
    std::fesetround(rounding);
    check(rejected_rounding);
    rows.codes[0] = -128;
    rejects([&] { q::pack(rows); });
    rows.codes[0] = 0;
    rows.scales[0] = 0;
    rejects([&] { q::pack(rows); });
    rows.scales[0] = 1;
    rows.inputs = 255;
    rejects([&] { q::pack(rows); });
    // Every W2 tile must retain its row scale while moving into K-major order.
    q::Rows value;
    value.outputs = 2048;
    value.inputs = 8192;
    value.codes.resize(value.outputs * value.inputs);
    value.scales.resize(value.outputs);
    for (size_t row = 0; row < value.outputs; ++row) {
      value.scales[row] = float(row + 1);
      for (size_t col = 0; col < value.inputs; ++col)
        value.codes[row * value.inputs + col] = int8_t((row + col / 256) % 127);
    }
    const auto stream = q::pack_value_k_major(value);
    for (size_t worker = 0; worker < 4; ++worker)
      for (size_t k = 0; k < 32; ++k)
        for (size_t row_tile = 0; row_tile < 32; ++row_tile) {
          const size_t tile =
              (worker * 1024 + k * 32 + row_tile) * q::tile_bytes;
          for (size_t lane = 0; lane < 16; ++lane) {
            const size_t row = worker * 512 + row_tile * 16 + lane;
            check(int8_t(stream[tile + lane * 256]) == int8_t((row + k) % 127));
            float scale;
            std::memcpy(&scale, stream.data() + tile + 4096 + lane * 4, 4);
            check(scale == value.scales[row]);
          }
        }
    value.outputs = 16;
    rejects([&] { q::pack_value_k_major(value); });
    std::cout << "INT8 signs/ties/scales/transpose/packing/invalid-input "
                 "checks passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

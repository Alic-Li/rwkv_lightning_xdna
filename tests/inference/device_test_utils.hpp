// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "rwkv/xdna/session.hpp"
#include <cstring>
#include <stdexcept>
namespace rwkv::xdna::test {
inline uint16_t bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  return static_cast<uint16_t>((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
inline float expand(uint16_t x) {
  uint32_t u = uint32_t(x) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
struct Guarded {
  DeviceBuffer root, data;
  size_t bytes;
  Guarded(Session &s, size_t n)
      : root(s.allocate(n + 128)), data(root.slice(64, n)), bytes(n) {
    std::vector<uint8_t> poison(n + 128, 0xcd);
    root.upload(poison.data(), poison.size());
  }
  void guard() {
    std::vector<uint8_t> v(bytes + 128);
    root.download(v.data(), v.size());
    for (size_t i = 0; i < 64; ++i)
      if (v[i] != 0xcd || v[bytes + 64 + i] != 0xcd)
        throw std::runtime_error("guard overwritten");
  }
};
} // namespace rwkv::xdna::test

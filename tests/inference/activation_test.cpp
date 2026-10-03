// SPDX-License-Identifier: Apache-2.0
#include "device_test_utils.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
using namespace rwkv::xdna;
using namespace rwkv::xdna::test;
static float from_bits(uint32_t bits) {
  float value; std::memcpy(&value, &bits, 4); return value;
}
int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: activation-test KERNEL_ROOT");
    auto root = std::filesystem::path(argv[1]) / "activation-test";
    Session session(root / "design.xclbin", root / "instructions.bin");
    Guarded input(session, 2048 * 4), output(session, 4096 * 4);
    auto run = session.prepare({input.data, output.data});
    std::vector<float> x(2048), y(4096), same(2048);
    size_t checked = 0;
    auto check = [&] {
      input.data.upload(x.data(), input.bytes);
      run.execute();
      output.data.download(y.data(), output.bytes);
      for (size_t i = 0; i < x.size(); ++i) {
        const double v = x[i] > 0 ? double(x[i]) : 0;
        const float reference = float(v * v);
        if (std::memcmp(&y[i], &y[2048+i], 4) ||
            std::memcmp(&reference, &y[2048+i], 4))
          throw std::runtime_error("Square scalar/vector/FP64 oracle bits differ at case " +
                                   std::to_string(checked + i));
      }
      input.data.download(same.data(), input.bytes);
      if (std::memcmp(x.data(), same.data(), input.bytes))
        throw std::runtime_error("Activation input overwritten");
      input.guard(); output.guard(); checked += x.size();
    };
    const uint32_t special[] = {0, 0x80000000u, 0x7f800000u, 0xff800000u,
        0x7fc00000u, 0xffc00000u, 1, 0x80000001u};
    for (size_t i = 0; i < x.size(); ++i) x[i] = from_bits(special[i % 8]);
    check();
    // Exhaust all 24-bit normal significands at exponent zero.
    for (uint32_t base = 0; base < (1u << 23); base += 2048) {
      for (uint32_t i = 0; i < 2048; ++i) x[i] = from_bits(0x3f800000u + base + i);
      check();
    }
    // Cover finite exponent/sign patterns, including subnormal and overflow.
    std::mt19937 rng(20261003);
    for (int batch = 0; batch < 128; ++batch) {
      for (auto &value : x) {
        uint32_t bits = rng();
        if ((bits & 0x7f800000u) == 0x7f800000u) bits ^= 0x00800000u;
        value = from_bits(bits);
      }
      check();
    }
    for (uint32_t exponent = 0; exponent < 255; ++exponent) {
      for (uint32_t i = 0; i < 2048; ++i)
        x[i] = from_bits((exponent << 23) | (i < 1024 ? i : 0x7fffffu - (i - 1024)));
      check();
    }
    std::cout << "activation scalar/vector/FP64 bits and guards passed values=" << checked << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}

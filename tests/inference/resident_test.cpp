// SPDX-License-Identifier: Apache-2.0
#include "rwkv/xdna/session.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace rwkv::xdna;
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("kernel directory required");
    std::filesystem::path root(argv[1]);
    Session ops(root / "resident-ops/design.xclbin",
                root / "resident-ops/instructions.bin");
    auto make = [&](size_t n, float value) {
      auto b = ops.allocate(n * 4);
      std::vector<float> v(n, value);
      b.upload(v.data(), n * 4);
      return b;
    };
    auto meta = make(16, 0), x = make(2048, 0), zero = make(2048, 0);
    auto middle = make(2048, 0), guarded = make(2048 + 32, 12345);
    auto output = guarded.slice(32, (2048 + 16) * 4).slice(32, 2048 * 4);
    std::vector<float> m(16, 0);
    m[0] = 0;
    m[1] = 2048;
    m[2] = 1;
    meta.upload(m.data(), m.size() * 4);
    auto first = ops.prepare({meta, x, x, zero, zero, middle});
    auto second = ops.prepare({meta, middle, middle, zero, zero, output});
    for (int pass = 1; pass <= 3; ++pass) {
      std::vector<float> input(2048, float(pass)), result(2080);
      x.upload(input.data(), input.size() * 4);
      first.execute();
      second.execute();
      guarded.download(result.data(), result.size() * 4);
      for (size_t i = 0; i < result.size(); ++i)
        if (result[i] != (i < 16 || i >= 2064 ? 12345.f : 4.f * pass))
          throw std::runtime_error("Resident chain numerical/guard failure");
    }
    bool rejected = false;
    try {
      x.slice(x.size() - 1, 4);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("Invalid slice accepted");
    std::cout << "Resident ops chain, replay and guards passed" << std::endl;
    Session wkv(root / "resident-decode/design.xclbin",
                root / "resident-decode/instructions.bin");
    auto state_guard = make(4096 + 32, 12345), vectors = make(6 * 2048, 0);
    auto y_guard = make(64 + 32, 12345);
    auto state = state_guard.slice(64, 4096 * 4), y = y_guard.slice(64, 64 * 4);
    std::vector<float> packed(6 * 2048, 0);
    for (size_t i = 0; i < 4; ++i) {
      std::fill_n(packed.begin() + i * 2048, 64, 1.f);
      std::fill_n(packed.begin() + i * 2048 + 31 * 64, 64, i == 1 ? 1.f : 2.f);
    }
    vectors.upload(packed.data(), packed.size() * 4);
    for (size_t head : {size_t(0), size_t(31)}) {
      std::vector<float> zeros(4096, 0);
      state.upload(zeros.data(), zeros.size() * 4);
      auto step =
          wkv.prepare({state, vectors.slice(head * 64 * 4, 10304 * 4), y});
      for (int pass = 1; pass <= 2; ++pass) {
        step.execute();
        std::vector<float> result(96), matrix(4128);
        y_guard.download(result.data(), result.size() * 4);
        state_guard.download(matrix.data(), matrix.size() * 4);
        for (size_t i = 0; i < result.size(); ++i)
          if (result[i] !=
              (i < 16 || i >= 80 ? 12345.f : pass * (head ? 512.f : 64.f)))
            throw std::runtime_error("WKV output/guard mismatch");
        for (size_t i = 0; i < matrix.size(); ++i)
          if (matrix[i] !=
              (i < 16 || i >= 4112 ? 12345.f : pass * (head ? 4.f : 1.f)))
            throw std::runtime_error("WKV state/guard mismatch");
      }
    }
    std::cout << "Resident recurrent passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

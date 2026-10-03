// SPDX-License-Identifier: Apache-2.0
#include "device_test_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
using namespace rwkv::xdna;
using namespace rwkv::xdna::test;
using V = std::vector<float>;
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("usage: projection-residual-test KERNEL_ROOT");
    auto session = [](std::filesystem::path root, std::string name) {
      return Session(root / name / "design.xclbin",
                     root / name / "instructions.bin");
    };
    auto fused = session(argv[1], "bf16-projection-residual");
    Guarded x(fused, 2048 * 4), w(fused, 4194304 * 2), res(fused, 2048 * 4),
        out(fused, 4096 * 4);
    auto run = fused.prepare({x.data, w.data, res.data, out.data});
    std::mt19937 rng(192);
    std::uniform_real_distribution<float> d(-.25f, .25f);
    std::vector<uint16_t> weights(4194304);
    for (auto &v : weights)
      v = bf16(d(rng));
    w.data.upload(weights.data(), w.bytes);
    V input(2048), r(2048), actual(4096);
    double worst = 0;
    for (int pass = 0; pass < 3; ++pass) {
      for (auto &v : input)
        v = d(rng) * (pass + 1);
      for (auto &v : r)
        v = d(rng);
      x.data.upload(input.data(), x.bytes);
      res.data.upload(r.data(), res.bytes);
      run.execute();
      out.data.download(actual.data(), out.bytes);
      auto check = [&](float actual, double ref) {
        double e = std::abs(double(actual) - ref);
        worst = std::max(worst, e);
        if (!std::isfinite(actual) || e > 2e-5 + 2e-5 * std::abs(ref))
          throw std::runtime_error("independent oracle mismatch");
      };
      for (size_t row = 0; row < 2048; ++row) {
        double sum = 0;
        for (size_t col = 0; col < 2048; ++col) {
          size_t pos = ((row / 16) * 8 + col / 256) * 4096 + (row % 16) * 256 +
                       col % 256;
          sum += double(expand(weights[pos])) * expand(bf16(input[col]));
        }
        check(actual[row], sum);
        check(actual[2048 + row], actual[row] + r[row]);
      }
    }
    for (auto *p : {&x, &w, &res, &out})
      p->guard();
    std::cout << "projection/residual oracle/replay/guards passed max_abs="
              << worst << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

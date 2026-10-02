// SPDX-License-Identifier: Apache-2.0
#include "rwkv/inference/model.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace rwkv::inference;
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("artifact path required");
    auto cpu = cpu_backend(), npu = full_npu_backend(argv[1]);
    std::mt19937 rng(243);
    std::uniform_real_distribution<float> rand(-1, 1);
    double worst = 0;
    auto check = [&](const Vector &actual, const Vector &want) {
      if (actual.size() != want.size())
        throw std::runtime_error("size mismatch");
      for (size_t i = 0; i < want.size(); ++i) {
        double d = std::abs(actual[i] - want[i]);
        worst = std::max(worst, d);
        if (!std::isfinite(actual[i]) || d > 2e-5 + 2e-4 * std::abs(want[i]))
          throw std::runtime_error("Mismatch at " + std::to_string(i) +
                                   " actual=" + std::to_string(actual[i]) +
                                   " expected=" + std::to_string(want[i]));
      }
    };
    Vector x(2048), y(2048), z(2048), w(2048);
    for (auto *v : {&x, &y, &z, &w})
      for (float &f : *v)
        f = rand(rng);
    for (int op = 0; op <= 12; ++op) {
      std::cout << "op " << op << std::endl;
      size_t group = op >= 9 && op <= 11 ? 64 : 1;
      check(npu->element(Op(op), x, y, z, w, group, 1e-5),
            cpu->element(Op(op), x, y, z, w, group, 1e-5));
    }
    // Saturation, near-zero nonlinearities and constant normalization.
    for (size_t i = 0; i < x.size(); ++i)
      x[i] = float(int(i % 101) - 50);
    y.assign(x.size(), 0);
    z.assign(x.size(), 0);
    w.assign(x.size(), 0);
    for (Op op : {Op::Tanh, Op::Sigmoid, Op::Decay})
      check(npu->element(op, x, y), cpu->element(op, x, y));
    x.assign(2048, 0.125f);
    y.assign(2048, 1);
    z.assign(2048, 0.01f);
    check(npu->element(Op::Norm, x, y, z, {}, 2048, 1e-5),
          cpu->element(Op::Norm, x, y, z, {}, 2048, 1e-5));
    bool rejected = false;
    try {
      npu->element(Op::Add, x);
    } catch (const std::exception &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("Missing operand accepted");
    for (size_t k : {256, 2048, 8192}) {
      x.resize(k);
      for (auto &f : x)
        f = rand(rng);
      Tensor weight{{272, k}, Vector(272 * k)};
      for (auto &f : weight.data)
        f = rand(rng) * 0.1f;
      std::cout << "gemv " << k << std::endl;
      check(npu->linear(x, weight), cpu->linear(x, weight));
    }
    std::cout << "All NPU primitive checks passed, max_abs_error=" << worst
              << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

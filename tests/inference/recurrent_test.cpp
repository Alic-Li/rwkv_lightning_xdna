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
    auto backend = argc > 1 ? npu_backend(argv[1]) : cpu_backend();
    constexpr size_t n = 64, heads = 2, c = n * heads;
    std::mt19937 rng(928);
    std::uniform_real_distribution<float> dist(-0.15f, 0.15f);
    Vector state(c * n), r(c), w(c), k(c), v(c), a(c), b(c);
    for (auto &s : state)
      s = dist(rng);
    std::vector<double> expected(state.begin(), state.end());
    double max_error = 0;
    for (int step = 0; step < 64; ++step) {
      for (size_t i = 0; i < c; ++i) {
        r[i] = dist(rng);
        w[i] = 0.9f + dist(rng) * 0.3f;
        k[i] = dist(rng);
        v[i] = dist(rng);
        a[i] = dist(rng);
        b[i] = dist(rng);
      }
      Vector reference(c);
      for (size_t h = 0; h < heads; ++h) {
        std::vector<double> sa(n);
        for (size_t i = 0; i < n; ++i)
          for (size_t j = 0; j < n; ++j)
            sa[j] += double(a[h * n + i]) * expected[(h * n + i) * n + j];
        for (size_t i = 0; i < n; ++i)
          for (size_t j = 0; j < n; ++j)
            expected[(h * n + i) * n + j] =
                expected[(h * n + i) * n + j] * w[h * n + i] +
                sa[j] * b[h * n + i] + double(k[h * n + i]) * v[h * n + j];
        for (size_t j = 0; j < n; ++j) {
          double out = 0;
          for (size_t i = 0; i < n; ++i)
            out += double(r[h * n + i]) * expected[(h * n + i) * n + j];
          reference[h * n + j] = float(out);
        }
      }
      auto y = backend->step(state, r, w, k, v, a, b, n);
      auto check = [&](double actual, double desired) {
        const double error = std::abs(actual - desired);
        max_error = std::max(max_error, error);
        if (!std::isfinite(actual) || error > 2e-6 + 2e-5 * std::abs(desired))
          throw std::runtime_error(
              "Recurrent FP64 reference mismatch at step " +
              std::to_string(step));
      };
      for (size_t i = 0; i < c; ++i)
        check(y[i], reference[i]);
      for (size_t i = 0; i < c * n; ++i)
        check(state[i], expected[i]);
    }
    // No hidden device state: zero state, zero k must produce exactly zero.
    state.assign(c * n, 0);
    k.assign(c, 0);
    auto y = backend->step(state, r, w, k, v, a, b, n);
    for (float f : y)
      if (f != 0)
        throw std::runtime_error("Reset leaked state");
    bool rejected = false;
    try {
      Vector wrong;
      backend->step(wrong, r, w, k, v, a, b, n);
    } catch (const std::exception &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("Invalid state accepted");
    std::cout << "64 transitions, 2 heads, reset and shape validation passed; "
                 "max_abs_error="
              << max_error << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

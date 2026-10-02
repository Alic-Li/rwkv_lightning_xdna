// SPDX-License-Identifier: Apache-2.0
#include "rwkv/inference/graph.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace rwkv::inference;
using Clock = std::chrono::steady_clock;
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("weights and kernel directory required");
    Weights weights(argv[1]);
    auto backend = full_npu_backend(argv[2]);
    Model model(weights, *backend);
    DecodeGraph baseline(weights, *backend);
    auto start = Clock::now();
    DecodeGraph resident(weights, *backend, argv[2]);
    std::cout << "resident_prepare_seconds="
              << std::chrono::duration<double>(Clock::now() - start).count()
              << std::endl;
    const auto stats = resident.stats();
    std::cout << "runs=" << stats.device_runs
              << " resident_bytes=" << stats.resident_bytes
              << " upload_bytes=" << stats.replay_upload_bytes
              << " download_bytes=" << stats.replay_download_bytes << std::endl;
    auto a = model.initial_state(), b = a;
    double worst = 0, baseline_total = 0, resident_total = 0;
    auto check = [&](const Vector &x, const Vector &y) {
      if (x.size() != y.size())
        throw std::runtime_error("shape mismatch");
      for (size_t i = 0; i < x.size(); ++i) {
        double error = std::abs(double(x[i]) - y[i]);
        worst = std::max(worst, error);
        if (!std::isfinite(x[i]) || !std::isfinite(y[i]) ||
            error > 2e-6 + 2e-5 * std::abs(y[i]))
          throw std::runtime_error("Resident logits/state mismatch");
      }
    };
    int iteration = 0;
    for (int token : {1, 2, 7, 9}) {
      start = Clock::now();
      auto expected = baseline.replay(token, a);
      double ordinary =
          std::chrono::duration<double>(Clock::now() - start).count();
      start = Clock::now();
      auto actual = resident.replay(token, b);
      double optimized =
          std::chrono::duration<double>(Clock::now() - start).count();
      check(actual, expected);
      for (size_t i = 0; i < a.layers.size(); ++i) {
        check(a.layers[i].matrix, b.layers[i].matrix);
        check(a.layers[i].attention_shift, b.layers[i].attention_shift);
        check(a.layers[i].ffn_shift, b.layers[i].ffn_shift);
      }
      std::cout << "token=" << token << " baseline_seconds=" << ordinary
                << " resident_seconds=" << optimized
                << " max_abs_error=" << worst << std::endl;
      if (iteration++) {
        baseline_total += ordinary;
        resident_total += optimized;
      }
    }
    std::cout << "measured_tokens=3 baseline_seconds=" << baseline_total
              << " resident_seconds=" << resident_total
              << " speedup=" << baseline_total / resident_total
              << " resident_tokens_per_second=" << 3 / resident_total
              << std::endl;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

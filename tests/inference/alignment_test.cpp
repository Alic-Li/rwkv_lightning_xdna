// SPDX-License-Identifier: Apache-2.0
#include "rwkv/inference/graph.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace rwkv::inference;
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("weights and kernels required");
#ifdef _OPENMP
    omp_set_num_threads(8);
#endif
    Weights w(argv[1]);
    auto cpu = cpu_backend(), npu = full_npu_backend(argv[2]);
    Model model(w, *cpu);
    DecodeGraph reference(w, *cpu), actual(w, *npu, argv[2]);
    std::vector<Vector> outputs(reference.stats().nodes),
        matrices(outputs.size());
    reference.set_trace([&](size_t i, const Vector &x, const Vector *s) {
      outputs.at(i) = x;
      if (s)
        matrices.at(i) = *s;
    });
    double max_abs = 0, max_scaled = 0;
    auto check = [&](const Vector &x, const Vector &y, size_t node) {
      if (x.size() != y.size())
        throw std::runtime_error("shape mismatch");
      double err = 0, scale = 0, sq = 0, refsq = 0;
      for (size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i]))
          throw std::runtime_error("nonfinite");
        err = std::max(err, std::abs(double(x[i]) - y[i]));
        scale = std::max(scale, std::abs(double(y[i])));
        sq += std::pow(double(x[i]) - y[i], 2);
        refsq += double(y[i]) * y[i];
      }
      max_abs = std::max(max_abs, err);
      max_scaled = std::max(max_scaled, err / std::max(1., scale));
      // FP32 serial CPU reduction vs AIE vector tree: norm-scaled error, plus
      // L2.
      if (err > 2e-5 + 2e-4 * scale ||
          std::sqrt(sq) > 2e-5 * std::sqrt(x.size()) + 2e-4 * std::sqrt(refsq))
        throw std::runtime_error("alignment node=" + std::to_string(node) +
                                 " abs=" + std::to_string(err) +
                                 " scale=" + std::to_string(scale));
    };
    actual.set_trace([&](size_t i, const Vector &x, const Vector *s) {
      check(x, outputs.at(i), i);
      if (s)
        check(*s, matrices.at(i), i);
    });
    auto a = model.initial_state(), b = a;
    int token = 1;
    for (int t = 0; t < 8; ++t) {
      auto x = reference.replay(token, a), y = actual.replay(token, b);
      check(y, x, outputs.size());
      auto greedy = [](const Vector &v) {
        return std::max_element(v.begin(), v.end()) - v.begin();
      };
      if (greedy(x) != greedy(y))
        throw std::runtime_error("greedy mismatch");
      for (size_t l = 0; l < a.layers.size(); ++l) {
        check(b.layers[l].matrix, a.layers[l].matrix, l);
        check(b.layers[l].attention_shift, a.layers[l].attention_shift, l);
        check(b.layers[l].ffn_shift, a.layers[l].ffn_shift, l);
      }
      token = t < 4 ? std::vector<int>{2, 7, 9, 3}[t] : int(greedy(x));
      std::cout << "step=" << t << " next_input=" << token
                << " greedy=" << greedy(x) << " all_nodes=" << outputs.size()
                << " max_abs=" << max_abs << " max_scaled=" << max_scaled
                << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

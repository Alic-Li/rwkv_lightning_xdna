// SPDX-License-Identifier: Apache-2.0
#include "rwkv/inference/graph.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace rwkv::inference;
// Independent CPU oracle for the BF16 main-projection contract. Low-rank
// projections remain FP32. Double scalar dots do not mirror the AIE tree.
class Bf16Reference : public RecurrentBackend {
  std::unique_ptr<RecurrentBackend> cpu = cpu_backend();
  static float rounded(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000;
    std::memcpy(&x, &u, 4);
    return x;
  }

public:
  Vector linear(const Vector &x, const Tensor &w, bool transposed) override {
    if (transposed)
      return cpu->linear(x, w, true);
    Vector y(w.shape[0]), input(x);
    for (auto &f : input)
      f = rounded(f);
#pragma omp parallel for
    for (size_t i = 0; i < y.size(); ++i) {
      double sum = 0;
      for (size_t j = 0; j < x.size(); ++j)
        sum += double(input[j]) * rounded(w.data[i * x.size() + j]);
      y[i] = float(sum);
    }
    return y;
  }
  Vector step(Vector &s, const Vector &r, const Vector &d, const Vector &k,
              const Vector &v, const Vector &a, const Vector &b,
              size_t n) override {
    return cpu->step(s, r, d, k, v, a, b, n);
  }
};
int main(int argc, char **argv) {
  try {
    if (argc != 3 && argc != 4)
      throw std::runtime_error("weights and kernels required");
#ifdef _OPENMP
    omp_set_num_threads(8);
#endif
    Weights w(argv[1]);
    auto cpu = cpu_backend(), npu = full_npu_backend(argv[2]);
    if (argc == 4) {
      if (std::string(argv[3]) != "--bf16" &&
          std::string(argv[3]) != "--bf16-projections")
        throw std::runtime_error("unknown option");
      setenv("RWKV_XDNA_BF16", "1", 1);
      cpu = std::make_unique<Bf16Reference>();
    }
    Model model(w, *cpu);
    DecodeGraph reference(w, *cpu), actual(w, *npu, argv[2]);
    if (argc == 4 && std::string(argv[3]) == "--bf16-projections") {
      size_t checked = 0;
      double worst = 0;
      actual.set_projection_trace([&](size_t node, const Vector &input,
                                      const Tensor &weight,
                                      const Vector &output) {
        auto expected = cpu->linear(input, weight, false);
        for (size_t i = 0; i < output.size(); ++i) {
          double e = std::abs(double(output[i]) - expected[i]);
          worst = std::max(worst, e);
          // Existing main projection unit-test threshold. The reference uses
          // the actual consumed input, isolating computation from earlier
          // BF16 rounding-boundary differences in independent trajectories.
          if (!std::isfinite(output[i]) ||
              e > 2e-5 + 2e-5 * std::abs(expected[i]))
            throw std::runtime_error(
                "projection oracle node=" + std::to_string(node) +
                " index=" + std::to_string(i) + " error=" + std::to_string(e));
        }
        ++checked;
      });
      actual.load_state(model.initial_state());
      for (int token : {1, 2, 7, 9})
        actual.replay_resident(token);
      std::cout << "BF16 real-input projection oracle passed projections="
                << checked << " max_abs=" << worst << std::endl;
      return 0;
    }
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

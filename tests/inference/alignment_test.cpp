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
// Independent CPU oracle for all BF16 projections. Double scalar dots do not
// mirror the AIE tree.
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
    Vector y(w.shape[transposed ? 1 : 0]), input(x);
    for (auto &f : input)
      f = rounded(f);
#pragma omp parallel for
    for (size_t i = 0; i < y.size(); ++i) {
      double sum = 0;
      for (size_t j = 0; j < x.size(); ++j)
        sum +=
            double(input[j]) *
            rounded(w.data[transposed ? j * y.size() + i : i * x.size() + j]);
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
    if (argc != 3)
      throw std::runtime_error("weights and kernels required");
#ifdef _OPENMP
    omp_set_num_threads(8);
#endif
    Weights w(argv[1]);
    auto cpu = std::make_unique<Bf16Reference>();
    Model model(w, *cpu);
    DecodeGraph actual(w, argv[2]);
    {
      size_t checked = 0;
      double worst = 0;
      actual.set_projection_trace([&](size_t node, const Vector &input,
                                      const Tensor &weight, bool transposed,
                                      const Vector &output) {
        auto expected = cpu->linear(input, weight, transposed);
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
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

// SPDX-License-Identifier: Apache-2.0
#include "rwkv/inference/model.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace rwkv::inference;
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("weights and kernel directory required");
    Weights w(argv[1]);
    auto backend = full_npu_backend(argv[2]);
    Model m(w, *backend);
    double worst = 0;
    auto check = [&](const Vector &a, const Vector &b) {
      if (a.size() != b.size())
        throw std::runtime_error("size mismatch");
      for (size_t i = 0; i < a.size(); ++i) {
        double e = std::abs(a[i] - b[i]);
        worst = std::max(worst, e);
        if (!std::isfinite(a[i]) || e > 2e-6 + 2e-5 * std::abs(b[i]))
          throw std::runtime_error("Prefill/decode discrepancy");
      }
    };
    for (size_t len : {1, 15, 16, 17, 33}) {
      auto a = m.initial_state(), b = m.initial_state();
      // Nonzero initial state and shift test: seed each request using decode.
      m.forward(3, a);
      m.forward(3, b);
      std::vector<int> ids;
      for (size_t i = 0; i < len; ++i)
        ids.push_back(int(i % 37 + 1));
      auto rows = m.prefill(ids, a);
      for (size_t i = 0; i < len; ++i)
        check(rows[i], m.forward(ids[i], b));
      for (size_t i = 0; i < a.layers.size(); ++i) {
        check(a.layers[i].matrix, b.layers[i].matrix);
        check(a.layers[i].attention_shift, b.layers[i].attention_shift);
        check(a.layers[i].ffn_shift, b.layers[i].ffn_shift);
      }
      check(m.forward(9, a), m.forward(9, b));
      std::cout << "prefill length " << len
                << ": logits, state, decode continuation passed\n";
    }
    std::cout << "max_abs_error=" << worst << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

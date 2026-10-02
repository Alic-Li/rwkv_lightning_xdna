// SPDX-License-Identifier: Apache-2.0
#include "rwkv/inference/graph.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace rwkv::inference;
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("weights and kernels required");
    Weights w(argv[1]);
    auto backend = full_npu_backend(argv[2]);
    Model model(w, *backend);
    DecodeGraph reference(w, *backend);
    std::vector<Vector> outputs(reference.stats().nodes),
        states(outputs.size());
    reference.set_trace([&](size_t i, const Vector &v, const Vector *s) {
      outputs[i] = v;
      if (s)
        states[i] = *s;
    });
    auto initial = model.initial_state(), state = initial;
    auto expected = reference.replay(1, state);
    backend->release_device_cache();
    DecodeGraph actual(w, *backend, argv[2]);
    size_t bad = 0;
    auto check = [&](size_t node, const Vector &v, const Vector &r,
                     const char *kind) {
      if (v.size() != r.size())
        throw std::runtime_error("trace shape mismatch");
      double abs = 0, ratio = 0;
      size_t worst = 0;
      for (size_t i = 0; i < v.size(); ++i) {
        if (!std::isfinite(v[i]) || !std::isfinite(r[i]))
          throw std::runtime_error("nonfinite trace");
        double err = std::abs(double(v[i]) - r[i]);
        abs = std::max(abs, err);
        double rel = err / (2e-6 + 2e-5 * std::abs(r[i]));
        if (rel > ratio) {
          ratio = rel;
          worst = i;
        }
      }
      if (ratio > 1) {
        ++bad;
        std::cout << "node=" << node << " " << kind << " max_abs=" << abs
                  << " tolerance_ratio=" << ratio << " worst=" << worst
                  << " actual=" << v[worst] << " reference=" << r[worst]
                  << std::endl;
      }
    };
    actual.set_trace([&](size_t i, const Vector &v, const Vector *s) {
      check(i, v, outputs[i], "output");
      if (s)
        check(i, *s, states[i], "state");
    });
    actual.replay(1, initial);
    std::cout << "failed_node_checks=" << bad << std::endl;
    return bad ? 1 : 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

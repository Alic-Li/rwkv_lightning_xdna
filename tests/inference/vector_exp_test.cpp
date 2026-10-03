// SPDX-License-Identifier: Apache-2.0
#include "device_test_utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
using namespace rwkv::xdna;
using namespace rwkv::xdna::test;
int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: vector-exp-test KERNEL_ROOT");
    const auto root = std::filesystem::path(argv[1]) / "vector-exp-test";
    Session session(root / "design.xclbin", root / "instructions.bin");
    Guarded input(session, 64 * 4), output(session, 256 * 4);
    auto run = session.prepare({input.data, output.data});
    std::vector<float> cases;
    auto around = [&](float x) {
      cases.push_back(std::nextafter(x, -std::numeric_limits<float>::infinity()));
      cases.push_back(x);
      cases.push_back(std::nextafter(x, std::numeric_limits<float>::infinity()));
    };
    for (float x : {0.f, -0.f, 1e-30f, 1e-12f, .01f, 1.f, 69.f, 70.f,
                    88.f, 100.f, 104.f, 1e10f, 1e30f}) {
      around(x); around(-x);
    }
    for (int k = -150; k <= 150; ++k) around((k + .5f) / 1.4426950408889634f);
    std::mt19937 rng(20261003);
    std::uniform_real_distribution<float> uniform(-105.f, 105.f);
    for (int i = 0; i < 8192; ++i) cases.push_back(uniform(rng));
    std::vector<float> x(64), y(256), same(64);
    double worst = 0;
    size_t checked = 0;
    for (size_t offset = 0; offset < cases.size(); offset += 64) {
      for (size_t i = 0; i < 64; ++i) x[i] = cases[(offset + i) % cases.size()];
      input.data.upload(x.data(), input.bytes);
      run.execute();
      output.data.download(y.data(), output.bytes);
      for (size_t i = 0; i < 64; ++i) {
        for (size_t base : {size_t(0), size_t(128)})
          if (std::memcmp(&y[base+i], &y[base+64+i], sizeof(float)))
            throw std::runtime_error("Scalar/vector bits differ for input=" + std::to_string(x[i]) + " slot=" + std::to_string(base));
        const double e = std::exp(-std::abs(double(x[i])));
        const double refs[] = {e, x[i] >= 0 ? 1 / (1 + e) : e / (1 + e)};
        for (size_t j = 0; j < 2; ++j) {
          const float value = y[j*128+64+i];
          double error = std::abs(double(value) - refs[j]);
          worst = std::max(worst, error);
          if (!std::isfinite(value) || error > 2e-5 + 2e-5 * std::abs(refs[j]))
            throw std::runtime_error("Independent transcendental oracle failed");
        }
        ++checked;
      }
      input.data.download(same.data(), input.bytes);
      if (same != x) throw std::runtime_error("Transcendental input overwritten");
    }
    input.guard(); output.guard();
    std::cout << "vector exponential/sigmoid scalar bits/oracle/guards passed values="
              << checked << " max_abs=" << worst << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}

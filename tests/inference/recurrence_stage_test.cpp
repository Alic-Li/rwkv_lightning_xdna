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
static void reference(V &state, V &aux) {
  auto sigmoid = [](double x) { return 1 / (1 + std::exp(-x)); };
  for (size_t h = 0; h < 32; ++h) {
    auto at = [&](size_t v, size_t j) -> float & {
      return aux[v * 2048 + h * 64 + j];
    };
    double ss = 0;
    for (size_t j = 0; j < 64; ++j) {
      float u = at(0, j) * at(3, j);
      ss += double(u) * u;
    }
    float den = std::max(float(std::sqrt(ss)), 1e-12f);
    for (size_t j = 0; j < 64; ++j) {
      float kk = at(0, j) * at(3, j) / den,
            a = float(sigmoid(at(1, j) + at(5, j)));
      at(19, j) = kk;
      at(20, j) = a;
      at(17, j) = -kk;
      at(18, j) = kk * a;
      at(15, j) = at(0, j) * (1 + (a - 1) * at(4, j));
      at(14, j) = float(
          std::exp(-0.6065306597126334f * float(sigmoid(at(2, j) + at(6, j)))));
    }
    for (size_t j = 0; j < 64; ++j) {
      double sa = 0, y = 0;
      for (size_t i = 0; i < 64; ++i)
        sa += double(at(17, i)) * state[h * 4096 + i * 64 + j];
      for (size_t i = 0; i < 64; ++i) {
        auto &s = state[h * 4096 + i * 64 + j];
        s = float(double(s) * at(14, i) + double(at(18, i)) * sa +
                  double(at(15, i)) * at(16, j));
        y += double(at(13, i)) * s;
      }
      at(8, j) = float(y);
    }
    double mean = 0, var = 0, dot = 0;
    for (size_t j = 0; j < 64; ++j)
      mean += at(8, j);
    mean /= 64;
    for (size_t j = 0; j < 64; ++j) {
      double d = at(8, j) - mean;
      var += d * d;
      dot += double(at(13, j)) * at(15, j) * at(11, j);
    }
    double scale = 1 / std::sqrt(var / 64 + 64e-5f);
    for (size_t j = 0; j < 64; ++j) {
      at(21, j) = float((at(8, j) - mean) * scale) * at(9, j) + at(10, j);
      at(22, j) = float(dot * at(16, j));
      at(23, j) = at(21, j) + at(22, j);
      at(24, j) = at(23, j) * at(12, j);
    }
  }
}
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("usage: recurrence-stage-test KERNEL_ROOT");
    auto session = [](std::filesystem::path root, const char *name) {
      return Session(root / name / "design.xclbin",
                     root / name / "instructions.bin");
    };
    auto fused = session(argv[1], "fused-recurrence-stage");
    Guarded fs(fused, 131072 * 4), fa(fused, 55296 * 4);
    auto run = fused.prepare({fs.data, fa.data});
    V state(131072), aux(55296), ref, expected, actual_s(state.size()),
        actual_a(aux.size());
    std::mt19937 rng(20261003);
    std::uniform_real_distribution<float> d(-.125f, .125f);
    for (auto &f : state)
      f = d(rng);
    for (auto &f : aux)
      f = d(rng);
    fs.data.upload(state.data(), fs.bytes);
    double worst = 0;
    for (int pass = 0; pass < 3; ++pass) {
      for (int v : {0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16})
        for (size_t i = 0; i < 2048; ++i)
          aux[v * 2048 + i] = (v == 9 ? 1.f : 0.f) + d(rng) * (pass + 1);
      expected = aux;
      ref = state;
      reference(ref, expected);
      fa.data.upload(aux.data(), fa.bytes);
      run.execute();
      fs.data.download(actual_s.data(), fs.bytes);
      fa.data.download(actual_a.data(), fa.bytes);
      auto check = [&](const V &a, const V &b) {
        for (size_t i = 0; i < a.size(); ++i) {
          double e = std::abs(double(a[i]) - b[i]);
          worst = std::max(worst, e);
          if (!std::isfinite(a[i]) || e > 2e-5 + 2e-5 * std::abs(b[i]))
            throw std::runtime_error(
                "independent oracle failed index=" + std::to_string(i) +
                " error=" + std::to_string(e));
        }
      };
      check(actual_s, ref);
      check(actual_a, expected);
      state = actual_s;
      aux = actual_a;
    }
    fs.guard();
    fa.guard();
    std::cout << "recurrence oracle/replay/guards passed max_abs=" << worst
              << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

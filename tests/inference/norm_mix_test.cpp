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
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("usage: norm-mix-test BASE_ROOT NEW_ROOT");
    auto session = [](std::filesystem::path root, std::string name) {
      return Session(root / name / "design.xclbin",
                     root / name / "instructions.bin");
    };
    for (size_t count : {1, 6}) {
      auto norm = session(argv[1], "upstream-norm"),
           mix = session(argv[1], count == 1 ? "fused-shift-mix" : "fused-mix"),
           fused = session(argv[2], "fused-norm-mix-" + std::to_string(count));
      Guarded x(norm, 2048 * 4), params(norm, (2 + count) * 2048 * 4),
          old(norm, 2048 * 4), base_old(norm, 2048 * 4), pair(norm, 4096 * 4),
          base_norm(norm, 2048 * 4), out(norm, count * 2048 * 4),
          base_out(norm, count * 2048 * 4);
      auto normrun =
          norm.prepare({x.data, params.data.slice(0, 2048 * 4),
                        params.data.slice(2048 * 4, 2048 * 4), base_norm.data});
      auto mixrun = mix.prepare({base_norm.data, base_old.data,
                                 params.data.slice(4096 * 4, count * 2048 * 4),
                                 base_out.data});
      auto run =
          fused.prepare({x.data, params.data, old.data, pair.data, out.data});
      std::mt19937 rng(571);
      std::uniform_real_distribution<float> d(-.5f, .5f);
      V input(2048), constants((2 + count) * 2048), previous(2048),
          normalized(2048), mixed(count * 2048), actual(count * 2048),
          baseline(count * 2048), npair(4096), shift(2048);
      for (auto &v : constants)
        v = d(rng);
      for (size_t i = 0; i < 2048; ++i)
        constants[i] += 1;
      for (auto &v : previous)
        v = d(rng);
      old.data.upload(previous.data(), old.bytes);
      base_old.data.upload(previous.data(), base_old.bytes);
      params.data.upload(constants.data(), params.bytes);
      double worst = 0;
      for (int pass = 0; pass < 3; ++pass) {
        for (auto &v : input)
          v = pass == 2 ? 100000.f : d(rng) * (pass + 1);
        double mean = 0, var = 0;
        for (auto v : input)
          mean += v;
        mean /= 2048;
        for (auto v : input)
          var += (v - mean) * (v - mean);
        for (size_t i = 0; i < 2048; ++i)
          normalized[i] =
              float((input[i] - mean) / std::sqrt(var / 2048 + 1e-5f)) *
                  constants[i] +
              constants[2048 + i];
        for (size_t p = 0; p < count; ++p)
          for (size_t i = 0; i < 2048; ++i)
            mixed[p * 2048 + i] =
                normalized[i] +
                (previous[i] - normalized[i]) * constants[(2 + p) * 2048 + i];
        x.data.upload(input.data(), x.bytes);
        normrun.execute();
        mixrun.execute();
        run.execute();
        out.data.download(actual.data(), out.bytes);
        base_out.data.download(baseline.data(), base_out.bytes);
        pair.data.download(npair.data(), pair.bytes);
        old.data.download(shift.data(), old.bytes);
        if (actual != baseline)
          throw std::runtime_error("fused/unfused mismatch");
        auto check = [&](float a, float b) {
          double e = std::abs(double(a) - b);
          worst = std::max(worst, e);
          if (!std::isfinite(a) || e > 2e-5 + 2e-5 * std::abs(b))
            throw std::runtime_error("independent oracle failed");
        };
        for (size_t i = 0; i < 2048; ++i) {
          check(npair[i], normalized[i]);
          if (npair[2048 + i] != previous[i] || shift[i] != npair[i])
            throw std::runtime_error("shift/pair mismatch");
        }
        for (size_t i = 0; i < mixed.size(); ++i)
          check(actual[i], mixed[i]);
        previous = shift;
      }
      auto measure = [&](bool combined) {
        std::vector<double> us;
        for (int i = 0; i < 50; ++i) {
          auto start = std::chrono::steady_clock::now();
          if (combined)
            run.execute();
          else {
            normrun.execute();
            mixrun.execute();
          }
          if (i >= 20)
            us.push_back(std::chrono::duration<double, std::micro>(
                             std::chrono::steady_clock::now() - start)
                             .count());
        }
        std::sort(us.begin(), us.end());
        return us[15];
      };
      double before = measure(false), after = measure(true);
      x.guard();
      params.guard();
      old.guard();
      base_old.guard();
      pair.guard();
      base_norm.guard();
      out.guard();
      base_out.guard();
      V same(constants.size());
      params.data.download(same.data(), params.bytes);
      if (same != constants)
        throw std::runtime_error("parameters overwritten");
      V same_input(input.size());
      x.data.download(same_input.data(), x.bytes);
      if (same_input != input)
        throw std::runtime_error("input overwritten");
      std::cout
          << "norm_mix=" << count
          << " exact same-precision/oracle/shift/replay/guards passed max_abs="
          << worst << " before_median_us=" << before
          << " after_median_us=" << after << '\n';
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

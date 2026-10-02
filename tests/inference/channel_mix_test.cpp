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
      throw std::runtime_error("usage: channel-mix-test BASE_ROOT NEW_ROOT");
    auto session = [](std::filesystem::path root, std::string name) {
      return Session(root / name / "design.xclbin",
                     root / name / "instructions.bin");
    };
    auto norm = session(argv[1], "fused-norm-mix-1"),
         ffn = session(argv[1], "bf16-ffn-pipeline"),
         fused = session(argv[2], "bf16-channel-mix");
    Guarded x(fused, 2048 * 4), params(fused, 6144 * 4),
        weights(fused, 33554432 * 2), diag(fused, 22528 * 4),
        out(fused, 4096 * 4), old(fused, 2048 * 4), npair(fused, 4096 * 4),
        mixed(fused, 2048 * 4), hidden(fused, 16384 * 4),
        baseline(fused, 4096 * 4);
    auto n =
        norm.prepare({x.data, params.data, old.data, npair.data, mixed.data});
    auto f = ffn.prepare(
        {mixed.data, weights.data, x.data, hidden.data, baseline.data});
    auto run =
        fused.prepare({x.data, params.data, weights.data, diag.data, out.data});
    std::mt19937 rng(730);
    std::uniform_real_distribution<float> d(-1, 1);
    std::vector<uint16_t> w(33554432);
    for (auto &v : w)
      v = bf16(d(rng) / 64);
    weights.data.upload(w.data(), weights.bytes);
    V input(2048), constants(6144), previous(2048), actual_diag(22528),
        actual(4096), base(4096), base_hidden(16384), base_pair(4096),
        base_mixed(2048);
    for (auto &v : constants)
      v = d(rng) / 8;
    for (size_t i = 0; i < 2048; ++i)
      constants[i] += 1;
    for (auto &v : previous)
      v = d(rng);
    params.data.upload(constants.data(), params.bytes);
    old.data.upload(previous.data(), old.bytes);
    diag.data.upload(previous.data(), old.bytes);
    auto pos = [](size_t row, size_t col, size_t K) {
      return ((row / 16) * (K / 256) + col / 256) * 4096 + (row % 16) * 256 +
             col % 256;
    };
    double worst = 0;
    auto check = [&](float actual, double ref) {
      double e = std::abs(double(actual) - ref);
      worst = std::max(worst, e);
      if (!std::isfinite(actual) || e > 2e-5 + 2e-5 * std::abs(ref))
        throw std::runtime_error("ChannelMix oracle error=" +
                                 std::to_string(e));
    };
    for (int pass = 0; pass < 3; ++pass) {
      for (auto &v : input)
        v = d(rng) * (pass + 1);
      x.data.upload(input.data(), x.bytes);
      n.execute();
      f.execute();
      run.execute();
      diag.data.download(actual_diag.data(), diag.bytes);
      out.data.download(actual.data(), out.bytes);
      baseline.data.download(base.data(), baseline.bytes);
      hidden.data.download(base_hidden.data(), hidden.bytes);
      npair.data.download(base_pair.data(), npair.bytes);
      mixed.data.download(base_mixed.data(), mixed.bytes);
      if (actual != base ||
          !std::equal(base_hidden.begin(), base_hidden.end(),
                      actual_diag.begin() + 6144) ||
          !std::equal(base_pair.begin(), base_pair.end(),
                      actual_diag.begin()) ||
          !std::equal(base_mixed.begin(), base_mixed.end(),
                      actual_diag.begin() + 4096))
        throw std::runtime_error("ChannelMix same-precision mismatch");
      double mean = 0, var = 0;
      for (auto v : input)
        mean += v;
      mean /= 2048;
      for (auto v : input)
        var += (v - mean) * (v - mean);
      for (size_t i = 0; i < 2048; ++i) {
        check(actual_diag[i],
              float((input[i] - mean) / std::sqrt(var / 2048 + 1e-5f)) *
                      constants[i] +
                  constants[2048 + i]);
        if (actual_diag[2048 + i] != previous[i])
          throw std::runtime_error("shift snapshot mismatch");
        check(actual_diag[4096 + i],
              actual_diag[i] +
                  (previous[i] - actual_diag[i]) * constants[4096 + i]);
        previous[i] = actual_diag[i];
      }
      for (size_t row = 0; row < 8192; ++row) {
        double sum = 0;
        for (size_t col = 0; col < 2048; ++col)
          sum += double(expand(w[pos(row, col, 2048)])) *
                 expand(bf16(actual_diag[4096 + col]));
        check(actual_diag[6144 + row], sum);
        float relu = std::max(actual_diag[6144 + row], 0.f);
        check(actual_diag[14336 + row], relu * relu);
      }
      for (size_t row = 0; row < 2048; ++row) {
        double sum = 0;
        for (size_t col = 0; col < 8192; ++col)
          sum += double(expand(w[16777216 + pos(row, col, 8192)])) *
                 expand(bf16(actual_diag[14336 + col]));
        check(actual[row], sum);
        check(actual[2048 + row], actual[row] + input[row]);
      }
    }
    auto measure = [&](int mode) {
      std::vector<double> us;
      for (int i = 0; i < 50; ++i) {
        auto start = std::chrono::steady_clock::now();
        if (mode == 1 || mode == 2)
          run.execute();
        if (mode == 0 || mode == 2 || mode == 3)
          n.execute();
        if (mode == 0)
          f.execute();
        if (i >= 20)
          us.push_back(std::chrono::duration<double, std::micro>(
                           std::chrono::steady_clock::now() - start)
                           .count());
      }
      std::sort(us.begin(), us.end());
      return std::make_pair(us.front(), us[15]);
    };
    auto before = measure(0), after = measure(1);
    const auto norm_only = measure(3), alternating = measure(2);
    for (auto *p : {&x, &params, &weights, &diag, &out, &old, &npair, &mixed,
                    &hidden, &baseline})
      p->guard();
    V ix(input.size()), cp(constants.size());
    x.data.download(ix.data(), x.bytes);
    params.data.download(cp.data(), params.bytes);
    std::vector<uint16_t> same(w.size());
    weights.data.download(same.data(), weights.bytes);
    if (ix != input || cp != constants || same != w)
      throw std::runtime_error("immutable input overwritten");
    std::cout << "ChannelMix same-precision/oracle/shift/replay/guards passed "
                 "max_abs="
              << worst << " before_min_us=" << before.first
              << " before_median_us=" << before.second
              << " after_min_us=" << after.first
              << " after_median_us=" << after.second << '\n';
    std::cout
        << "{\"case\":\"ChannelMix context alternation\",\"norm_median_us\":"
        << norm_only.second << ",\"channel_median_us\":" << after.second
        << ",\"alternating_pair_median_us\":" << alternating.second
        << ",\"alternation_excess_us\":"
        << alternating.second - norm_only.second - after.second
        << ",\"scope\":\"distinct programs, includes scheduling/PDI/cache "
           "effects\"}\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

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
      throw std::runtime_error("usage: ffn-pipeline-test BASE_ROOT NEW_ROOT");
    auto session = [](std::filesystem::path root, std::string name) {
      return Session(root / name / "design.xclbin",
                     root / name / "instructions.bin");
    };
    auto key = session(argv[1], "bf16-fused-ffn-key"),
         value = session(argv[1], "bf16-array-gemv-8192"),
         add = session(argv[1], "resident-ops-fast"),
         fused = session(argv[2], "bf16-ffn-pipeline");
    Guarded x(fused, 2048 * 4), w(fused, 33554432 * 2), res(fused, 2048 * 4),
        h(fused, 16384 * 4), out(fused, 4096 * 4), bh(fused, 16384 * 4),
        bo(fused, 4096 * 4), meta(fused, 16 * 4), zero(fused, 2048 * 4);
    std::vector<uint16_t> weights(33554432);
    std::mt19937 rng(537);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (auto &v : weights)
      v = bf16(d(rng) / 64);
    w.data.upload(weights.data(), w.bytes);
    V m(16, 0);
    m[1] = 2048;
    m[2] = 1;
    meta.data.upload(m.data(), meta.bytes);
    V zeros(2048, 0);
    zero.data.upload(zeros.data(), zero.bytes);
    auto k = key.prepare({x.data, w.data.slice(0, 16777216 * 2),
                          bh.data.slice(0, 8192 * 4),
                          bh.data.slice(8192 * 4, 8192 * 4)});
    auto v = value.prepare({bh.data.slice(8192 * 4, 8192 * 4),
                            w.data.slice(16777216 * 2, 16777216 * 2),
                            bo.data.slice(0, 2048 * 4)});
    auto a =
        add.prepare({meta.data, res.data, bo.data.slice(0, 2048 * 4), zero.data,
                     zero.data, bo.data.slice(2048 * 4, 2048 * 4)});
    auto run = fused.prepare({x.data, w.data, res.data, h.data, out.data});
    V input(2048), residual(2048), hidden(16384), baseline_h(16384),
        actual(4096), baseline(4096);
    auto pos = [](size_t row, size_t col, size_t K) {
      return ((row / 16) * (K / 256) + col / 256) * 4096 + (row % 16) * 256 +
             col % 256;
    };
    double worst = 0;
    auto check = [&](float actual, double ref) {
      double e = std::abs(double(actual) - ref);
      worst = std::max(worst, e);
      if (!std::isfinite(actual) || e > 2e-5 + 2e-5 * std::abs(ref))
        throw std::runtime_error("independent FFN oracle error=" +
                                 std::to_string(e));
    };
    for (int pass = 0; pass < 3; ++pass) {
      for (auto &f : input)
        f = d(rng) * (pass + 1);
      for (auto &f : residual)
        f = d(rng);
      x.data.upload(input.data(), x.bytes);
      res.data.upload(residual.data(), res.bytes);
      k.execute();
      v.execute();
      a.execute();
      run.execute();
      h.data.download(hidden.data(), h.bytes);
      bh.data.download(baseline_h.data(), bh.bytes);
      out.data.download(actual.data(), out.bytes);
      bo.data.download(baseline.data(), bo.bytes);
      if (hidden != baseline_h || actual != baseline)
        throw std::runtime_error("fused/unfused FFN mismatch");
      for (size_t row = 0; row < 8192; ++row) {
        double sum = 0;
        for (size_t col = 0; col < 2048; ++col)
          sum += double(expand(weights[pos(row, col, 2048)])) *
                 expand(bf16(input[col]));
        check(hidden[row], sum);
        float relu = std::max(hidden[row], 0.f);
        check(hidden[8192 + row], relu * relu);
      }
      for (size_t row = 0; row < 2048; ++row) {
        double sum = 0;
        for (size_t col = 0; col < 8192; ++col)
          sum += double(expand(weights[16777216 + pos(row, col, 8192)])) *
                 expand(bf16(hidden[8192 + col]));
        check(actual[row], sum);
        check(actual[2048 + row], actual[row] + residual[row]);
      }
    }
    auto measure = [&](bool combined) {
      std::vector<double> us;
      for (int i = 0; i < 50; ++i) {
        auto start = std::chrono::steady_clock::now();
        if (combined)
          run.execute();
        else {
          k.execute();
          v.execute();
          a.execute();
        }
        if (i >= 20)
          us.push_back(std::chrono::duration<double, std::micro>(
                           std::chrono::steady_clock::now() - start)
                           .count());
      }
      std::sort(us.begin(), us.end());
      return std::make_pair(us.front(), us[15]);
    };
    auto before = measure(false), after = measure(true);
    for (auto *g : {&x, &w, &res, &h, &out, &bh, &bo, &meta, &zero})
      g->guard();
    std::vector<uint16_t> same(weights.size());
    w.data.download(same.data(), w.bytes);
    if (same != weights)
      throw std::runtime_error("weights overwritten");
    V same_x(2048), same_r(2048);
    x.data.download(same_x.data(), x.bytes);
    res.data.download(same_r.data(), res.bytes);
    if (same_x != input || same_r != residual)
      throw std::runtime_error("inputs overwritten");
    std::cout
        << "FFN pipeline same-precision/oracle/replay/guards passed max_abs="
        << worst << " before_min_us=" << before.first
        << " before_median_us=" << before.second
        << " after_min_us=" << after.first
        << " after_median_us=" << after.second << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

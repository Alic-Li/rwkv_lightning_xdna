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
      throw std::runtime_error(
          "usage: projection-residual-test BASE_ROOT NEW_ROOT");
    auto session = [](std::filesystem::path root, std::string name) {
      return Session(root / name / "design.xclbin",
                     root / name / "instructions.bin");
    };
    auto gemv = session(argv[1], "bf16-array-gemv-2048"),
         add = session(argv[1], "resident-ops-fast"),
         fused = session(argv[2], "bf16-projection-residual");
    Guarded x(fused, 2048 * 4), w(fused, 4194304 * 2), res(fused, 2048 * 4),
        out(fused, 4096 * 4), base(fused, 4096 * 4), meta(fused, 16 * 4),
        zero(fused, 2048 * 4);
    auto g = gemv.prepare({x.data, w.data, base.data.slice(0, 2048 * 4)});
    auto a = add.prepare({meta.data, res.data, base.data.slice(0, 2048 * 4),
                          zero.data, zero.data,
                          base.data.slice(2048 * 4, 2048 * 4)});
    auto run = fused.prepare({x.data, w.data, res.data, out.data});
    std::mt19937 rng(192);
    std::uniform_real_distribution<float> d(-.25f, .25f);
    std::vector<uint16_t> weights(4194304);
    for (auto &v : weights)
      v = bf16(d(rng));
    w.data.upload(weights.data(), w.bytes);
    V m(16, 0);
    m[1] = 2048;
    m[2] = 1;
    meta.data.upload(m.data(), meta.bytes);
    V z(2048, 0);
    zero.data.upload(z.data(), zero.bytes);
    V input(2048), r(2048), actual(4096), baseline(4096);
    double worst = 0;
    for (int pass = 0; pass < 3; ++pass) {
      for (auto &v : input)
        v = d(rng) * (pass + 1);
      for (auto &v : r)
        v = d(rng);
      x.data.upload(input.data(), x.bytes);
      res.data.upload(r.data(), res.bytes);
      g.execute();
      a.execute();
      run.execute();
      out.data.download(actual.data(), out.bytes);
      base.data.download(baseline.data(), base.bytes);
      if (actual != baseline)
        throw std::runtime_error("fused/unfused mismatch");
      auto check = [&](float actual, double ref) {
        double e = std::abs(double(actual) - ref);
        worst = std::max(worst, e);
        if (!std::isfinite(actual) || e > 2e-5 + 2e-5 * std::abs(ref))
          throw std::runtime_error("independent oracle mismatch");
      };
      for (size_t row = 0; row < 2048; ++row) {
        double sum = 0;
        for (size_t col = 0; col < 2048; ++col) {
          size_t pos = ((row / 16) * 8 + col / 256) * 4096 + (row % 16) * 256 +
                       col % 256;
          sum += double(expand(weights[pos])) * expand(bf16(input[col]));
        }
        check(actual[row], sum);
        check(actual[2048 + row], actual[row] + r[row]);
      }
    }
    auto measure = [&](bool combined) {
      std::vector<double> us;
      for (int i = 0; i < 50; ++i) {
        auto start = std::chrono::steady_clock::now();
        if (combined)
          run.execute();
        else {
          g.execute();
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
    for (auto *p : {&x, &w, &res, &out, &base, &meta, &zero})
      p->guard();
    std::vector<uint16_t> same(weights.size());
    w.data.download(same.data(), w.bytes);
    if (same != weights)
      throw std::runtime_error("weight overwritten");
    V ix(2048), ir(2048);
    x.data.download(ix.data(), x.bytes);
    res.data.download(ir.data(), res.bytes);
    if (ix != input || ir != r)
      throw std::runtime_error("inputs overwritten");
    std::cout << "projection residual same-precision/oracle/replay/guards "
                 "passed max_abs="
              << worst << " before_min_us=" << before.first
              << " before_median_us=" << before.second
              << " after_min_us=" << after.first
              << " after_median_us=" << after.second << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

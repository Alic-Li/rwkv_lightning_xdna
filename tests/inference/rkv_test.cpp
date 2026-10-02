// SPDX-License-Identifier: Apache-2.0
#include "device_test_utils.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
using namespace rwkv::xdna;
using namespace rwkv::xdna::test;
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("kernel root required");
    auto p = std::filesystem::path(argv[1]) / "bf16-rkv";
    Session s(p / "design.xclbin", p / "instructions.bin");
    Guarded x(s, 6 * 2048 * 4), w(s, 3 * 2048 * 2048 * 2), r(s, 2048 * 4),
        k(s, 2048 * 4), v(s, 2048 * 4);
    std::vector<float> input(6 * 2048), reference(3 * 2048), output(3 * 2048);
    std::vector<uint16_t> weight(3 * 2048 * 2048), rowmajor(weight.size());
    std::mt19937 rng(191);
    std::uniform_real_distribution<float> dist(-0.125f, 0.125f);
    for (size_t proj = 0; proj < 3; ++proj)
      for (size_t row = 0; row < 2048; ++row)
        for (size_t col = 0; col < 2048; ++col) {
          const auto bits = bf16(dist(rng));
          rowmajor[(proj * 2048 + row) * 2048 + col] = bits;
          size_t pos =
              ((row / 256 * 3 + proj) * 16 + (row % 256) / 16) * 8 * 4096 +
              (col / 256) * 4096 + (row % 16) * 256 + col % 256;
          weight[pos] = bits;
        }
    w.data.upload(weight.data(), w.bytes);
    auto run = s.prepare({x.data, w.data, r.data, k.data, v.data});
    double worst = 0;
    for (int pass = 0; pass < 3; ++pass) {
      for (auto &f : input)
        f = dist(rng) * (pass + 1);
      x.data.upload(input.data(), x.bytes);
      for (size_t proj = 0; proj < 3; ++proj)
        for (size_t row = 0; row < 2048; ++row) {
          double dot = 0;
          const size_t slot = proj == 0 ? 0 : proj + 1;
          for (size_t col = 0; col < 2048; ++col)
            dot += double(expand(rowmajor[(proj * 2048 + row) * 2048 + col])) *
                   expand(bf16(input[slot * 2048 + col]));
          reference[proj * 2048 + row] = float(dot);
        }
      run.execute();
      r.data.download(output.data(), r.bytes);
      k.data.download(output.data() + 2048, k.bytes);
      v.data.download(output.data() + 4096, v.bytes);
      for (size_t i = 0; i < output.size(); ++i) {
        double e = std::abs(double(output[i]) - reference[i]);
        worst = std::max(worst, e);
        if (!std::isfinite(output[i]) ||
            e > 2e-5 + 2e-5 * std::abs(reference[i]))
          throw std::runtime_error("RKV mismatch index=" + std::to_string(i));
      }
    }
    std::vector<double> us;
    for (int i = 0; i < 50; ++i) {
      RunTiming t;
      run.execute(30000, &t);
      if (i >= 20)
        us.push_back(t.submit_us + t.wait_us);
    }
    x.guard();
    w.guard();
    r.guard();
    k.guard();
    v.guard();
    std::vector<float> xafter(input.size());
    x.data.download(xafter.data(), x.bytes);
    std::vector<uint16_t> wafter(weight.size());
    w.data.download(wafter.data(), w.bytes);
    if (xafter != input || wafter != weight)
      throw std::runtime_error("input overwritten");
    std::sort(us.begin(), us.end());
    std::cout << "RKV random oracle/replay/guards passed max_abs=" << worst
              << " min_us=" << us.front() << " median_us=" << us[us.size() / 2]
              << " max_us=" << us.back() << std::endl;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

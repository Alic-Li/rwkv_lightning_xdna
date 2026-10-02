// SPDX-License-Identifier: Apache-2.0
#include "device_test_utils.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
using namespace rwkv::xdna;
using namespace rwkv::xdna::test;
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("Usage: rwkv-gemm-probe-test ROOT");
    auto path = std::filesystem::path(argv[1]) / "bf16-gemm-probe";
    Session s(path / "design.xclbin", path / "instructions.bin");
    Guarded a(s, 8192), b(s, 8192), c(s, 16384);
    std::mt19937 rng(749);
    std::uniform_real_distribution<float> dist(-.25f, .25f);
    std::vector<uint16_t> ar(4096), br(4096), ap(4096), bp(4096);
    for (auto &v : ar)
      v = bf16(dist(rng));
    for (auto &v : br)
      v = bf16(dist(rng));
    for (size_t r = 0; r < 64; ++r)
      for (size_t k = 0; k < 64; ++k) {
        ap[((r / 4) * 8 + k / 8) * 32 + (r % 4) * 8 + k % 8] = ar[r * 64 + k];
        bp[((r / 8) * 8 + k / 8) * 64 + (r % 8) * 8 + k % 8] = br[r * 64 + k];
      }
    a.data.upload(ap.data(), a.bytes);
    b.data.upload(bp.data(), b.bytes);
    auto run = s.prepare({a.data, b.data, c.data});
    std::vector<double> times;
    double submit = 0, wait = 0;
    for (int i = 0; i < 50; ++i) {
      RunTiming t;
      run.execute(30000, &t);
      if (i >= 20) {
        times.push_back(t.submit_us + t.wait_us);
        submit += t.submit_us;
        wait += t.wait_us;
      }
    }
    std::vector<float> out(4096);
    c.data.download(out.data(), c.bytes);
    double max_abs = 0;
    for (size_t r = 0; r < 64; ++r)
      for (size_t n = 0; n < 64; ++n) {
        double ref = 0;
        for (size_t k = 0; k < 64; ++k)
          ref += double(expand(ar[r * 64 + k])) * expand(br[k * 64 + n]);
        float got = out[((r / 4) * 8 + n / 8) * 32 + (r % 4) * 8 + n % 8];
        double err = std::abs(got - ref);
        max_abs = std::max(max_abs, err);
        if (!std::isfinite(got) || err > 2e-5 + 2e-5 * std::abs(ref))
          throw std::runtime_error("GEMM numeric mismatch");
      }
    a.guard();
    b.guard();
    c.guard();
    std::vector<uint16_t> check(4096);
    a.data.download(check.data(), a.bytes);
    if (check != ap)
      throw std::runtime_error("A modified");
    b.data.download(check.data(), b.bytes);
    if (check != bp)
      throw std::runtime_error("B modified");
    std::sort(times.begin(), times.end());
    std::cout << nlohmann::json(
                     {{"operation", "GEMM"},
                      {"dtype", "BF16 native 4x8x8 MMUL, FP32 output"},
                      {"m", 64},
                      {"k", 64},
                      {"n", 64},
                      {"cores", 1},
                      {"warmups", 20},
                      {"samples", 30},
                      {"min_us", times.front()},
                      {"median_us", times[15]},
                      {"mean_submit_us", submit / 30},
                      {"mean_wait_us", wait / 30},
                      {"end_to_end_gflops",
                       2. * 64 * 64 * 64 / times.front() / 1000},
                      {"max_abs", max_abs},
                      {"guards", true},
                      {"scope", "prepared run includes DMA, scheduling, "
                                "compute and wait; not isolated core cycles"}})
                     .dump()
              << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

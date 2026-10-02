// SPDX-License-Identifier: Apache-2.0
#include "device_test_utils.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>
using namespace rwkv::xdna;
using namespace rwkv::xdna::test;
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("kernel root required");
    auto root = std::filesystem::path(argv[1]);
    Session gemm(root / "bf16-gemm-probe/design.xclbin",
                 root / "bf16-gemm-probe/instructions.bin");
    Session copy(root / "stream-group-3/design.xclbin",
                 root / "stream-group-3/instructions.bin");
    Guarded a(gemm, 8192), b(gemm, 8192), c(gemm, 16384), x(copy, 24576),
        y(copy, 24576);
    std::vector<uint16_t> ones(4096, bf16(1));
    std::vector<float> input(6144), out(6144), product(4096);
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = float(i) / 128;
    a.data.upload(ones.data(), a.bytes);
    b.data.upload(ones.data(), b.bytes);
    x.data.upload(input.data(), x.bytes);
    auto g = gemm.prepare({a.data, b.data, c.data});
    auto p = copy.prepare({x.data, y.data});
    auto measure = [&](int mode) {
      std::vector<double> times;
      for (int i = 0; i < 50; ++i) {
        auto start = std::chrono::steady_clock::now();
        if (mode != 1)
          g.execute();
        if (mode != 0)
          p.execute();
        if (i >= 20)
          times.push_back(std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - start)
                              .count());
      }
      std::sort(times.begin(), times.end());
      return times[15];
    };
    const double gm = measure(0), cp = measure(1), pair = measure(2);
    c.data.download(product.data(), c.bytes);
    y.data.download(out.data(), y.bytes);
    for (float f : product)
      if (f != 64)
        throw std::runtime_error("GEMM replay failed");
    if (out != input)
      throw std::runtime_error("copy replay failed");
    a.guard();
    b.guard();
    c.guard();
    x.guard();
    y.guard();
    std::cout << nlohmann::json(
                     {{"warmups", 20},
                      {"samples", 30},
                      {"gemm_median_us", gm},
                      {"copy_median_us", cp},
                      {"alternating_pair_median_us", pair},
                      {"alternation_excess_us", pair - gm - cp},
                      {"guards", true},
                      {"scope", "two distinct XCLBIN/hardware contexts; excess "
                                "includes scheduling/PDI/cache effects, not "
                                "isolated context switch latency"}})
                     .dump()
              << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

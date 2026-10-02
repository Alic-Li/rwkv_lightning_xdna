// SPDX-License-Identifier: Apache-2.0
#include "device_test_utils.hpp"
#include <algorithm>
#include <iostream>
#include <random>
using namespace rwkv::xdna;
using namespace rwkv::xdna::test;
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("kernel root required");
    for (size_t count : {3, 4}) {
      auto p = std::filesystem::path(argv[1]) /
               ("stream-group-" + std::to_string(count));
      Session s(p / "design.xclbin", p / "instructions.bin");
      Guarded x(s, count * 2048 * 4), y(s, count * 2048 * 4);
      auto run = s.prepare({x.data, y.data});
      std::vector<float> input(count * 2048), output(input.size()),
          same(input.size());
      std::mt19937 rng(194);
      std::uniform_real_distribution<float> d(-1, 1);
      for (int pass = 0; pass < 3; ++pass) {
        for (auto &f : input)
          f = d(rng);
        x.data.upload(input.data(), x.bytes);
        run.execute();
        y.data.download(output.data(), y.bytes);
        if (output != input)
          throw std::runtime_error("regroup data mismatch");
      }
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
      x.guard();
      y.guard();
      x.data.download(same.data(), x.bytes);
      if (same != input)
        throw std::runtime_error("source overwritten");
      std::sort(times.begin(), times.end());
      std::cout << "{\"case\":\"stream-group-" << count
                << "\",\"status\":\"passed\",\"bytes_each_direction\":"
                << x.bytes << ",\"warmup\":20,\"repetitions\":30,\"min_us\":"
                << times.front() << ",\"median_us\":" << times[15]
                << ",\"mean_submit_us\":" << submit / 30
                << ",\"mean_wait_us\":" << wait / 30
                << ",\"scope\":\"two identity workers, regroup DMA, prepared "
                   "submission and wait\"}"
                << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

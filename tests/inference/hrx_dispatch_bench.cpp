// SPDX-License-Identifier: Apache-2.0
#include "hrx_test_utils.hpp"
using namespace hrx_test;
constexpr int steps = 16, samples = 128;

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error(
          "Usage: rwkv-hrx-dispatch-bench NORM_ARTIFACT_DIR");
    const std::filesystem::path path(argv[1]);
    HrxProgram hrx(path);
    HrxBuffer a(hrx.stream), b(hrx.stream), g(hrx.stream), bt(hrx.stream);
    rwkv::xdna::Session xrt(path / "design.xclbin", path / "instructions.bin");
    XrtBuffer xa(xrt), xb(xrt), xg(xrt), xbt(xrt);
    auto even = xrt.prepare({xa.data, xg.data, xbt.data, xb.data});
    auto odd = xrt.prepare({xb.data, xg.data, xbt.data, xa.data});
    Vec gamma(channels), beta(channels);
    for (size_t i = 0; i < channels; ++i) {
      gamma[i] = .9f + float(i % 17) * .005f;
      beta[i] = float(int(i % 13) - 6) * .002f;
    }
    g.upload(gamma);
    bt.upload(beta);
    xg.upload(gamma);
    xbt.upload(beta);
    std::vector<Vec> fixtures;
    for (int fixture = 0; fixture < 4; ++fixture) {
      Vec input(channels);
      for (size_t i = 0; i < channels; ++i)
        input[i] = fixture == 0
                       ? .125f
                       : std::sin(float(i) * (.123f + .017f * fixture)) *
                                 (fixture == 1 ? .00001f : .2f) +
                             .125f;
      fixtures.push_back(input);
    }
    std::array<Vec, 4> xrt_results;
    double max_abs_fp64 = 0;
    for (size_t f = 0; f < fixtures.size(); ++f) {
      xa.upload(fixtures[f]);
      auto current = fixtures[f];
      // Validate each transition against FP64 using its actual FP32 input.
      // Repeated near-constant normalization amplifies earlier rounding errors;
      // an independently evolved FP64 trajectory tests a different contract.
      for (int i = 0; i < steps; ++i) {
        auto expected = reference(current, gamma, beta);
        (i % 2 ? odd : even).execute();
        current = (i % 2 ? xa : xb).download();
        for (size_t j = 0; j < channels; ++j) {
          const double error = std::abs(double(current[j]) - expected[j]);
          max_abs_fp64 = std::max(max_abs_fp64, error);
          if (!std::isfinite(current[j]) || error > 2e-5)
            throw std::runtime_error("Per-step FP64 reference mismatch");
        }
      }
      xrt_results[f] = current;
    }
    for (int trial = 0; trial < 4; ++trial) {
      // Reverse order every other trial; each backend gets four warm samples.
      for (int index = 0; index < 3; ++index) {
        const int mode = trial % 2 ? 2 - index : index;
        double elapsed = 0;
        for (int sample = -4; sample < samples; ++sample) {
          const size_t f = size_t(sample + 4) % fixtures.size();
          if (mode == 0)
            xa.upload(fixtures[f]);
          else
            a.upload(fixtures[f]);
          auto start = Clock::now();
          for (int step = 0; step < steps; ++step) {
            if (mode == 0)
              (step % 2 ? odd : even).execute();
            else {
              hrx.record({(step % 2 ? b : a).ref(), g.ref(), bt.ref(),
                          (step % 2 ? a : b).ref()});
              if (mode == 1 || step == steps - 1)
                hrx.finish();
            }
          }
          const double us =
              std::chrono::duration<double, std::micro>(Clock::now() - start)
                  .count();
          if (sample >= 0)
            elapsed += us;
          auto result = mode == 0 ? xa.download() : a.download();
          if (result != xrt_results[f])
            throw std::runtime_error("XRT/HRX mismatch");
          if (mode == 0) {
            xb.download();
            if (xg.download() != gamma || xbt.download() != beta)
              throw std::runtime_error("XRT immutable buffer changed");
          } else {
            b.download();
            if (g.download() != gamma || bt.download() != beta)
              throw std::runtime_error("HRX immutable buffer changed");
          }
        }
        std::cout << nlohmann::json({{"trial", trial},
                                     {"mode", mode == 0   ? "xrt_individual"
                                              : mode == 1 ? "hrx_individual"
                                                          : "hrx_batch16"},
                                     {"chain16_us", elapsed / samples},
                                     {"samples", samples},
                                     {"max_abs_fp64_per_step", max_abs_fp64},
                                     {"bitwise_xrt", true},
                                     {"guards", "pass"}})
                         .dump()
                  << '\n';
      }
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

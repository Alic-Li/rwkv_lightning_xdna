// SPDX-License-Identifier: Apache-2.0
// Actual norm output feeds projection; measures heterogeneous program dispatch.
#include "device_test_utils.hpp"
#include "hrx_test_utils.hpp"
using namespace hrx_test;
using rwkv::xdna::test::bf16;
using rwkv::xdna::test::expand;

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("Usage: rwkv-hrx-alternation-bench KERNEL_ROOT");
    auto root = std::filesystem::path(argv[1]);
    HrxProgram hn(root / "upstream-norm"),
        hp(root / "bf16-projection-residual");
    rwkv::xdna::Session xn(root / "upstream-norm/design.xclbin",
                           root / "upstream-norm/instructions.bin"),
        xp(root / "bf16-projection-residual/design.xclbin",
           root / "bf16-projection-residual/instructions.bin");
    HrxBuffer input(hn.stream), gamma(hn.stream), beta(hn.stream),
        norm(hn.stream), weights(hn.stream, 8388608), residual(hn.stream),
        output(hn.stream, 16384);
    XrtBuffer xi(xn), xg(xn), xb(xn), xy(xn), xw(xp, 8388608), xr(xp),
        xo(xp, 16384);
    Vec g(channels), b(channels), r(channels), w(2097152);
    std::vector<uint16_t> packed(4194304);
    for (size_t i = 0; i < channels; ++i) {
      g[i] = .9f + (i % 17) * .005f;
      b[i] = (int(i % 13) - 6) * .002f;
      r[i] = (int(i % 19) - 9) * .01f;
    }
    for (size_t row = 0; row < channels; ++row)
      for (size_t col = 0; col < channels; ++col) {
        size_t pos =
            ((row / 16) * 8 + col / 256) * 4096 + row % 16 * 256 + col % 256;
        packed[pos] =
            bf16(float(int((row * 31 + col * 17) % 101) - 50) * .0005f);
      }
    std::memcpy(w.data(), packed.data(), 8388608);
    gamma.upload(g);
    beta.upload(b);
    residual.upload(r);
    weights.upload(w);
    xg.upload(g);
    xb.upload(b);
    xr.upload(r);
    xw.upload(w);
    auto nr = xn.prepare({xi.data, xg.data, xb.data, xy.data});
    auto pr = xp.prepare({xy.data, xw.data, xr.data, xo.data});
    std::vector<Vec> inputs, norms, outputs;
    double max_norm = 0, max_proj = 0;
    for (int f = 0; f < 4; ++f) {
      Vec in(channels);
      for (size_t i = 0; i < channels; ++i)
        in[i] = f == 0 ? .125f
                       : .125f + std::sin(float(i) * (.123f + f * .017f)) *
                                     (f == 1 ? .00001f : .2f);
      xi.upload(in);
      nr.execute();
      auto n = xy.download();
      auto ref = reference(in, g, b);
      for (size_t i = 0; i < channels; ++i) {
        double e = std::abs(double(n[i]) - ref[i]);
        max_norm = std::max(max_norm, e);
        if (!std::isfinite(n[i]) || e > 2e-5)
          throw std::runtime_error("norm reference");
      }
      pr.execute();
      auto out = xo.download();
      for (size_t row = 0; row < channels; ++row) {
        double sum = 0;
        for (size_t col = 0; col < channels; ++col) {
          size_t pos =
              ((row / 16) * 8 + col / 256) * 4096 + row % 16 * 256 + col % 256;
          sum += double(expand(packed[pos])) * expand(bf16(n[col]));
        }
        for (size_t part = 0; part < 2; ++part) {
          double ref = sum + (part ? r[row] : 0),
                 e = std::abs(out[part * channels + row] - ref);
          max_proj = std::max(max_proj, e);
          if (!std::isfinite(out[part * channels + row]) ||
              e > 2e-5 + 2e-5 * std::abs(ref))
            throw std::runtime_error("projection reference");
        }
      }
      inputs.push_back(in);
      norms.push_back(n);
      outputs.push_back(out);
    }
    for (int trial = 0; trial < 4; ++trial)
      for (int j = 0; j < 3; ++j) {
        int mode = trial % 2 ? 2 - j : j;
        double elapsed = 0;
        for (int sample = -4; sample < 64; ++sample) {
          size_t f = (sample + 4) % 4;
          if (mode == 0)
            xi.upload(inputs[f]);
          else
            input.upload(inputs[f]);
          auto start = Clock::now();
          if (mode == 0) {
            nr.execute();
            pr.execute();
          } else {
            hn.record({input.ref(), gamma.ref(), beta.ref(), norm.ref()});
            if (mode == 1)
              hn.finish();
            hp.record({norm.ref(), weights.ref(), residual.ref(), output.ref()},
                      hn.stream);
            hn.finish();
          }
          double us =
              std::chrono::duration<double, std::micro>(Clock::now() - start)
                  .count();
          if (sample >= 0)
            elapsed += us;
          auto n = mode == 0 ? xy.download() : norm.download(),
               out = mode == 0 ? xo.download() : output.download();
          if (n != norms[f] || out != outputs[f])
            throw std::runtime_error("XRT HRX result mismatch");
          if (mode == 0) {
            if (xi.download() != inputs[f] || xg.download() != g ||
                xb.download() != b || xr.download() != r)
              throw std::runtime_error("XRT immutable");
          } else if (input.download() != inputs[f] || gamma.download() != g ||
                     beta.download() != b || residual.download() != r)
            throw std::runtime_error("HRX immutable");
        }
        if (std::memcmp(weights.download().data(), w.data(), 8388608) ||
            std::memcmp(xw.download().data(), w.data(), 8388608))
          throw std::runtime_error("weights changed");
        std::cout << nlohmann::json({{"trial", trial},
                                     {"mode", mode == 0   ? "xrt_individual"
                                              : mode == 1 ? "hrx_individual"
                                                          : "hrx_pair"},
                                     {"pair_us", elapsed / 64},
                                     {"samples", 64},
                                     {"max_norm_fp64", max_norm},
                                     {"max_proj_fp64", max_proj},
                                     {"bitwise_xrt", true},
                                     {"guards", "pass"}})
                         .dump()
                  << std::endl;
      }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}

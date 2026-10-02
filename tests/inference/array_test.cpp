// SPDX-License-Identifier: Apache-2.0
#include "rwkv/xdna/session.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace rwkv::xdna;
struct Guarded {
  DeviceBuffer root, data;
  size_t n;
  Guarded(Session &s, size_t count)
      : root(s.allocate((count + 32) * 4)), data(root.slice(64, count * 4)),
        n(count) {
    std::vector<float> v(n + 32, 12345);
    root.upload(v.data(), v.size() * 4);
  }
  void upload(const std::vector<float> &v) {
    if (v.size() != n)
      throw std::runtime_error("input shape");
    data.upload(v.data(), n * 4);
  }
  std::vector<float> read() {
    std::vector<float> v(n + 32);
    root.download(v.data(), v.size() * 4);
    for (size_t i = 0; i < 16; ++i)
      if (v[i] != 12345 || v[n + 16 + i] != 12345)
        throw std::runtime_error("guard overwritten");
    return {v.begin() + 16, v.end() - 16};
  }
};
void check(const std::vector<float> &a, const std::vector<float> &b,
           float atol = 2e-5f) {
  if (a.size() != b.size())
    throw std::runtime_error("shape");
  for (size_t i = 0; i < a.size(); ++i)
    if (!std::isfinite(a[i]) ||
        std::abs(a[i] - b[i]) > atol + 2e-5f * std::abs(b[i]))
      throw std::runtime_error("numerical element=" + std::to_string(i) +
                               " actual=" + std::to_string(a[i]) +
                               " expected=" + std::to_string(b[i]));
}
int main(int argc, char **argv) {
  try {
    if (argc != 2 && argc != 3)
      throw std::runtime_error("kernel root required");
    std::filesystem::path root(argv[1]);
    const bool array32 = argc == 3 && std::string(argv[2]) == "--array32";
    for (int mode = 0; mode < 5; ++mode) {
      size_t k = mode == 0   ? 256
                 : mode == 2 ? 8192
                             : 2048,
             rows = mode >= 3 ? 8192 : 2048;
      std::string name = mode == 4 ? "fused-ffn-key"
                                   : "array-gemv-" + std::to_string(k) +
                                         (mode == 3 ? "-8192" : "");
      if (array32 && mode < 4)
        name.replace(0, 5, "array32");
      Session s(root / name / "design.xclbin",
                root / name / "instructions.bin");
      Guarded x(s, k), w(s, rows * k), y(s, rows), z(s, rows);
      std::vector<float> weights(rows * k), input(k), expected(rows),
          active(rows);
      for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < k; ++c) {
          size_t pos = ((r / 16) * (k / 256) + c / 256) * 4096 +
                       (r % 16) * 256 + c % 256;
          if (array32 && mode < 4) {
            size_t stripe = rows / 32;
            pos = (((r / (stripe * 4) * (stripe / 16) + (r % stripe) / 16) *
                        (k / 256) +
                    c / 256) *
                       4 +
                   (r / stripe) % 4) *
                      4096 +
                  (r % 16) * 256 + c % 256;
          }
          weights[pos] = (int(r % 17) - 8) * (int(c % 7) - 3) / 2048.f;
        }
      w.upload(weights);
      std::vector<DeviceBuffer> args{x.data, w.data, y.data};
      if (mode == 4)
        args.push_back(z.data);
      auto run = s.prepare(args);
      for (int pass = 1; pass <= 3; ++pass) {
        for (size_t c = 0; c < k; ++c)
          input[c] = (int(c % 11) - 5) * pass / 16.f;
        x.upload(input);
        for (size_t r = 0; r < rows; ++r) {
          float sum = 0;
          for (size_t c = 0; c < k; ++c)
            sum += input[c] * (int(r % 17) - 8) * (int(c % 7) - 3) / 2048.f;
          expected[r] = sum;
          active[r] = std::pow(std::max(sum, 0.f), 2);
        }
        run.execute();
        check(y.read(), expected);
        if (mode == 4)
          check(z.read(), active);
        check(x.read(), input);
      }
      check(w.read(), weights);
      std::cout << name << " replay/numerical/guards passed" << std::endl;
    }
    {
      Session s(root / "upstream-norm/design.xclbin",
                root / "upstream-norm/instructions.bin");
      Guarded x(s, 2048), w(s, 2048), b(s, 2048), y(s, 2048);
      std::vector<float> in(2048), gamma(2048, 1.1f), beta(2048, 0.2f),
          expected(2048);
      w.upload(gamma);
      b.upload(beta);
      auto run = s.prepare({x.data, w.data, b.data, y.data});
      for (int pass = 0; pass < 3; ++pass) {
        for (size_t i = 0; i < in.size(); ++i)
          in[i] =
              pass == 2 ? 100000.f : float(int(i % 23) - 11) * (pass + 1) / 8;
        double mean = 0, var = 0;
        for (float f : in)
          mean += f;
        mean /= in.size();
        for (float f : in)
          var += (f - mean) * (f - mean);
        for (size_t i = 0; i < in.size(); ++i)
          expected[i] =
              float((in[i] - mean) / std::sqrt(var / in.size() + 1e-5f)) *
                  gamma[i] +
              beta[i];
        x.upload(in);
        run.execute();
        check(y.read(), expected);
        check(x.read(), in);
      }
      check(w.read(), gamma);
      check(b.read(), beta);
      std::cout << "upstream FP32 affine norm replay/numerical/guards passed"
                << std::endl;
    }
    {
      Session s(root / "fused-prepare/design.xclbin",
                root / "fused-prepare/instructions.bin");
      Guarded aux(s, 16384), rec(s, 12288), out(s, 4096);
      std::vector<float> input(16384, 0), state(12288, 0), expected(4096);
      for (size_t i = 0; i < 2048; ++i) {
        input[i] = 1;
        input[3 * 2048 + i] = 1;
        input[4 * 2048 + i] = 1;
        state[i] = 2;
        state[3 * 2048 + i] = 3;
        expected[i] = 0.125f;
        expected[2048 + i] = 0.5f;
      }
      aux.upload(input);
      rec.upload(state);
      auto run = s.prepare({aux.data, rec.data, out.data});
      for (int pass = 0; pass < 3; ++pass) {
        run.execute();
        check(out.read(), expected);
        for (size_t i = 0; i < 2048; ++i) {
          state[2048 + i] = std::exp(-0.6065306597126334f * 0.5f);
          state[2 * 2048 + i] = 0.5f;
          state[4 * 2048 + i] = -0.125f;
          state[5 * 2048 + i] = 0.0625f;
        }
        check(rec.read(), state);
        check(aux.read(), input);
      }
      std::cout << "fused preparation replay/numerical/guards passed"
                << std::endl;
    }
    {
      Session s(root / "fused-value/design.xclbin",
                root / "fused-value/instructions.bin");
      Guarded aux(s, 6144), first(s, 2048), out(s, 2048);
      std::vector<float> a(6144), f(2048), expected(2048);
      auto run = s.prepare({aux.data, first.data, out.data});
      for (int pass = 0; pass < 3; ++pass) {
        for (size_t i = 0; i < 2048; ++i) {
          a[i] = float(int(i % 31) - 15) / 8;
          a[2048 + i] = float(int(i % 23) - 11) * (pass + 1);
          a[4096 + i] = float(int(i % 7) - 3) / 4;
          f[i] = float(int(i % 19) - 9) / 3;
          float sigmoid =
              float(1. / (1. + std::exp(-double(a[2048 + i] + a[4096 + i]))));
          expected[i] = a[i] + (f[i] - a[i]) * sigmoid;
        }
        aux.upload(a);
        first.upload(f);
        run.execute();
        check(out.read(), expected);
        check(aux.read(), a);
        check(first.read(), f);
      }
      std::cout << "fused value replay/numerical/guards passed" << std::endl;
    }
    {
      Session s(root / "array-decode/design.xclbin",
                root / "array-decode/instructions.bin");
      Guarded state(s, 131072), packed(s, 12288), out(s, 2048);
      std::vector<float> ref(131072), p(12288), y(2048);
      for (size_t i = 0; i < ref.size(); ++i)
        ref[i] = float(int(i % 37) - 18) / 512;
      state.upload(ref);
      auto run = s.prepare({state.data, packed.data, out.data});
      for (int pass = 0; pass < 3; ++pass) {
        for (size_t i = 0; i < 2048; ++i) {
          p[i] = float(int(i % 7) - 3) / 32;
          p[2048 + i] = 0.7f + float(i % 11) / 64;
          p[4096 + i] = float(int(i % 13) - 6) / 64;
          p[6144 + i] = float(int(i % 17) - 8) * (pass + 1) / 64;
          p[8192 + i] = float(int(i % 19) - 9) / 128;
          p[10240 + i] = float(int(i % 23) - 11) / 128;
        }
        for (size_t h = 0; h < 32; ++h)
          for (size_t j = 0; j < 64; ++j) {
            float sa = 0, sum = 0;
            for (size_t i = 0; i < 64; ++i)
              sa += p[8192 + h * 64 + i] * ref[h * 4096 + i * 64 + j];
            for (size_t i = 0; i < 64; ++i) {
              auto index = h * 4096 + i * 64 + j;
              ref[index] = ref[index] * p[2048 + h * 64 + i] +
                           p[10240 + h * 64 + i] * sa +
                           p[4096 + h * 64 + i] * p[6144 + h * 64 + j];
              sum += p[h * 64 + i] * ref[index];
            }
            y[h * 64 + j] = sum;
          }
        packed.upload(p);
        run.execute();
        check(state.read(), ref, 2e-6f);
        check(out.read(), y, 2e-6f);
        check(packed.read(), p);
      }
      std::cout << "32-head resident WKV replay/numerical/guards passed"
                << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

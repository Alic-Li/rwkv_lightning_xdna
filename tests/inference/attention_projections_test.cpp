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
    if (argc != 2 && argc != 3)
      throw std::runtime_error("kernel root required");
    const std::vector<size_t> counts =
        argc == 3 ? std::vector<size_t>{std::stoul(argv[2])}
                  : std::vector<size_t>{3, 4};
    for (size_t count : counts) {
      auto path = std::filesystem::path(argv[1]) /
                  ("bf16-attention-projections-" + std::to_string(count));
      Session s(path / "design.xclbin", path / "instructions.bin");
      constexpr size_t rkv_size = 3 * 2048 * 2048;
      Guarded x(s, 12288 * 4),
          weights(s, (rkv_size + count * 2 * 256 * 2048) * 2),
          aux(s, count * 512 * 4), arena(s, 55296 * 4), value_aux(s, 6144 * 4);
      auto raw = aux.data.slice(0, count * 256 * 4),
           active = aux.data.slice(count * 256 * 4, count * 256 * 4);
      std::vector<DeviceBuffer> ys{arena.data.slice(4096 * 4, 2048 * 4),
                                   arena.data.slice(2048 * 4, 2048 * 4),
                                   arena.data.slice(12 * 2048 * 4, 2048 * 4),
                                   value_aux.data.slice(2048 * 4, 2048 * 4)};
      std::vector<DeviceBuffer> rkv_out{
          arena.data.slice(13 * 2048 * 4, 2048 * 4),
          arena.data.slice(0, 2048 * 4),
          count == 3 ? arena.data.slice(16 * 2048 * 4, 2048 * 4)
                     : value_aux.data.slice(0, 2048 * 4)};
      std::vector<float> input(12288), r(count * 256), a(r.size()),
          output(2048);
      std::vector<uint16_t> rw1(count * 256 * 2048), rw2(count * 2048 * 256),
          pw1(rw1.size()), pw2(rw2.size());
      std::mt19937 rng(591);
      std::uniform_real_distribution<float> d(-0.125f, 0.125f);
      for (size_t p = 0; p < count; ++p) {
        for (size_t row = 0; row < 256; ++row)
          for (size_t col = 0; col < 2048; ++col) {
            auto bits = bf16(d(rng));
            rw1[(p * 256 + row) * 2048 + col] = bits;
            size_t pos =
                ((row / 128 * count + p) * 8 + (row % 128) / 16) * 8 * 4096 +
                col / 256 * 4096 + (row % 16) * 256 + col % 256;
            pw1[pos] = bits;
          }
        for (size_t row = 0; row < 2048; ++row)
          for (size_t col = 0; col < 256; ++col) {
            auto bits = bf16(d(rng));
            rw2[(p * 2048 + row) * 256 + col] = bits;
            size_t pos =
                ((row / 512 * count + p) * 32 + (row % 512) / 16) * 4096 +
                (row % 16) * 256 + col;
            pw2[pos] = bits;
          }
      }
      std::vector<uint16_t> rkv_ref(rkv_size), packed(rkv_size);
      for (size_t p = 0; p < 3; ++p)
        for (size_t row = 0; row < 2048; ++row)
          for (size_t col = 0; col < 2048; ++col) {
            auto bits = bf16(d(rng));
            rkv_ref[(p * 2048 + row) * 2048 + col] = bits;
            size_t pos =
                ((row / 256 * 3 + p) * 16 + (row % 256) / 16) * 8 * 4096 +
                col / 256 * 4096 + (row % 16) * 256 + col % 256;
            packed[pos] = bits;
          }
      packed.insert(packed.end(), pw1.begin(), pw1.end());
      packed.insert(packed.end(), pw2.begin(), pw2.end());
      weights.data.upload(packed.data(), weights.bytes);
      std::vector<DeviceBuffer> args{x.data, weights.data, arena.data,
                                     value_aux.data, aux.data};
      auto run = s.prepare(args);
      double worst = 0;
      auto check = [&](float actual, double expected) {
        double e = std::abs(double(actual) - expected);
        worst = std::max(worst, e);
        if (!std::isfinite(actual) || e > 2e-5 + 2e-5 * std::abs(expected))
          throw std::runtime_error("attention projections numerical error=" +
                                   std::to_string(e));
      };
      for (int pass = 0; pass < 3; ++pass) {
        for (auto &f : input)
          f = d(rng) * (pass + 1);
        x.data.upload(input.data(), x.bytes);
        run.execute();
        for (size_t p = 0; p < 3; ++p) {
          rkv_out[p].download(output.data(), 2048 * 4);
          const int slots[] = {0, 2, 3};
          for (size_t row = 0; row < 2048; ++row) {
            double sum = 0;
            for (size_t col = 0; col < 2048; ++col)
              sum += double(expand(rkv_ref[(p * 2048 + row) * 2048 + col])) *
                     expand(bf16(input[slots[p] * 2048 + col]));
            check(output[row], sum);
          }
        }
        raw.download(r.data(), r.size() * 4);
        active.download(a.data(), a.size() * 4);
        for (size_t p = 0; p < count; ++p) {
          const int slots[] = {1, 4, 5, 3};
          for (size_t row = 0; row < 256; ++row) {
            double sum = 0;
            for (size_t col = 0; col < 2048; ++col)
              sum += double(expand(rw1[(p * 256 + row) * 2048 + col])) *
                     expand(bf16(input[slots[p] * 2048 + col]));
            check(r[p * 256 + row], sum);
            float v = r[p * 256 + row];
            double expected = p == 0   ? std::tanh(double(v))
                              : p == 2 ? 1 / (1 + std::exp(-double(v)))
                                       : v;
            check(a[p * 256 + row], expected);
          }
          ys[p].download(output.data(), 2048 * 4);
          for (size_t row = 0; row < 2048; ++row) {
            double sum = 0;
            for (size_t col = 0; col < 256; ++col)
              sum += double(expand(rw2[(p * 2048 + row) * 256 + col])) *
                     expand(bf16(a[p * 256 + col]));
            check(output[row], sum);
          }
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
      weights.guard();
      aux.guard();
      arena.guard();
      value_aux.guard();
      auto untouched = [&](Guarded &g, const std::vector<size_t> &slots) {
        std::vector<uint32_t> words(g.bytes / 4);
        g.data.download(words.data(), g.bytes);
        for (size_t i = 0; i < words.size(); ++i)
          if (std::find(slots.begin(), slots.end(), i / 2048) == slots.end() &&
              words[i] != 0xcdcdcdcd)
            throw std::runtime_error(
                "attention projection overwrote unused arena slot");
      };
      std::vector<size_t> written{0, 1, 2, 12, 13};
      if (count == 3)
        written.push_back(16);
      untouched(arena, written);
      untouched(value_aux,
                count == 4 ? std::vector<size_t>{0, 1} : std::vector<size_t>{});
      std::vector<float> ia(input.size());
      x.data.download(ia.data(), x.bytes);
      std::vector<uint16_t> after(packed.size());
      weights.data.download(after.data(), weights.bytes);
      if (ia != input || after != packed)
        throw std::runtime_error("input overwritten");
      std::sort(us.begin(), us.end());
      std::cout << "attention_projections=" << count
                << " oracle/replay/guards passed max_abs=" << worst
                << " min_us=" << us.front()
                << " median_us=" << us[us.size() / 2] << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

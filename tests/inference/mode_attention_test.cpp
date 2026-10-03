// SPDX-License-Identifier: Apache-2.0
#include "device_test_utils.hpp"
#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
using namespace rwkv::xdna;
using namespace rwkv::xdna::test;
using V = std::vector<float>;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error(
          "Usage: mode-attention-test CANDIDATE_ROOT BASELINE_ROOT");
    auto session = [](const char *root, const char *name) {
      auto p = std::filesystem::path(root) / name;
      return Session(p / "design.xclbin", p / "instructions.bin");
    };
    auto cn = session(argv[1], "mode-norm-mix-6"),
         ca = session(argv[1], "bf16-attention-projections-4"),
         bn = session(argv[2], "fused-norm-mix-6"),
         ba = session(argv[2], "bf16-attention-projections-4"),
         third = session(argv[2], "upstream-norm");
    Guarded x(cn, 8192), p(cn, 18432 * 4), w(ca, (16777216 + 4096) * 2),
        cs(cn, 8192), bs(bn, 8192), cp(cn, 16384), bp(bn, 16384),
        cm(cn, 12288 * 4), bm(bn, 12288 * 4), cy(ca, 55296 * 4),
        by(ba, 55296 * 4), cv(ca, 6144 * 4), bv(ba, 6144 * 4), cr(ca, 2048 * 4),
        br(ba, 2048 * 4), ty(third, 8192);
    V input(2048), params(18432, 0), old(2048);
    std::mt19937 rng(552);
    std::uniform_real_distribution<float> rnd(-.25f, .25f);
    for (size_t i = 0; i < 16384; ++i)
      params[i] = rnd(rng) + (i < 2048 ? 1.f : 0.f);
    p.data.upload(params.data(), p.bytes);
    std::vector<uint16_t> weights(16777216 + 4096, 0);
    for (size_t i = 0; i < 16777216; ++i)
      weights[i] = bf16(rnd(rng) / 16);
    weights[16777217] = 0x3f80;
    w.data.upload(weights.data(), w.bytes);
    auto nc = cn.prepare({x.data, p.data, cs.data, cp.data, cm.data});
    auto nb = bn.prepare(
        {x.data, p.data.slice(0, 16384 * 4), bs.data, bp.data, bm.data});
    auto ac = ca.prepare({cm.data, w.data, cy.data, cv.data, cr.data});
    auto ab = ba.prepare(
        {bm.data, w.data.slice(0, 16777216 * 2), by.data, bv.data, br.data});
    auto tr = third.prepare(
        {x.data, p.data.slice(0, 8192), p.data.slice(8192, 8192), ty.data});
    auto equal = [](Guarded &a, Guarded &b) {
      V av(a.bytes / 4), bv(b.bytes / 4);
      a.data.download(av.data(), a.bytes);
      b.data.download(bv.data(), b.bytes);
      if (std::memcmp(av.data(), bv.data(), a.bytes))
        throw std::runtime_error("Bitwise stage mismatch");
    };
    auto check = [&]() {
      equal(cs, bs);
      equal(cp, bp);
      equal(cm, bm);
      equal(cy, by);
      equal(cv, bv);
      equal(cr, br);
    };
    double max_norm = 0;
    for (int fixture = 0; fixture < 3; ++fixture) {
      for (size_t i = 0; i < 2048; ++i) {
        input[i] = fixture == 0
                       ? rnd(rng)
                       : (fixture == 1 ? .25f : .125f + rnd(rng) * 1e-4f);
        old[i] = rnd(rng);
      }
      x.data.upload(input.data(), x.bytes);
      cs.data.upload(old.data(), cs.bytes);
      bs.data.upload(old.data(), bs.bytes);
      for (auto *out : {&cp, &bp, &cm, &bm, &cy, &by, &cv, &bv, &cr, &br}) {
        V sentinel(out->bytes / 4, -777);
        out->data.upload(sentinel.data(), out->bytes);
      }
      nc.execute();
      ac.execute();
      nb.execute();
      ab.execute();
      check();
      V got(4096);
      cp.data.download(got.data(), cp.bytes);
      double mean = std::accumulate(input.begin(), input.end(), 0.) / 2048,
             var = 0;
      for (float v : input)
        var += (v - mean) * (v - mean) / 2048;
      for (size_t i = 0; i < 2048; ++i) {
        double ref = (input[i] - mean) / std::sqrt(var + 1e-5) * params[i] +
                     params[2048 + i],
               err = std::abs(got[i] - ref);
        max_norm = std::max(max_norm, err);
        if (!std::isfinite(got[i]) || err > 2e-5 + 2e-5 * std::abs(ref))
          throw std::runtime_error("Norm FP64 oracle mismatch");
      }
      // Consecutive mode-zero iterations must not consume matrix packets;
      // repeated mode-one iterations must not consume old-state/affine packets.
      for (int i = 0; i < 3; ++i) {
        nc.execute();
        nb.execute();
      }
      for (int i = 0; i < 2; ++i) {
        ac.execute();
        ab.execute();
      }
      check();
    }
    // Shift state is now steady for the final fixed input; both paths remain
    // equivalent.
    for (bool entry : {false, true})
      for (int arm : {0, 1, 1, 0}) {
        auto &n = arm ? nc : nb;
        auto &a = arm ? ac : ab;
        for (int i = 0; i < 16; ++i) {
          if (entry)
            tr.execute();
          n.execute();
          a.execute();
        }
        double nt = 0, at = 0;
        for (int i = 0; i < 128; ++i) {
          if (entry)
            tr.execute();
          auto t = Clock::now();
          n.execute();
          auto m = Clock::now();
          a.execute();
          auto e = Clock::now();
          nt += std::chrono::duration<double, std::micro>(m - t).count();
          at += std::chrono::duration<double, std::micro>(e - m).count();
        }
        check();
        std::cout << Json({{"candidate", bool(arm)},
                           {"after_third_program", entry},
                           {"samples", 128},
                           {"norm_mix_mean_us", nt / 128},
                           {"attention_mean_us", at / 128},
                           {"pair_mean_us", (nt + at) / 128}})
                         .dump()
                  << std::endl;
      }
    std::vector<uint16_t> saved_w(weights.size());
    w.data.download(saved_w.data(), w.bytes);
    if (saved_w != weights)
      throw std::runtime_error("Weights mutated");
    V saved_p(params.size());
    p.data.download(saved_p.data(), p.bytes);
    if (saved_p != params)
      throw std::runtime_error("Parameters mutated");
    V saved_x(input.size());
    x.data.download(saved_x.data(), x.bytes);
    if (saved_x != input)
      throw std::runtime_error("Input mutated");
    for (auto *b : {&x, &p, &w, &cs, &bs, &cp, &bp, &cm, &bm, &cy, &by, &cv,
                    &bv, &cr, &br, &ty})
      b->guard();
    std::cout << Json({{"status", "passed"},
                       {"fixtures", 3},
                       {"norm_fp64_max_abs", max_norm},
                       {"bitwise_state_outputs_immutable_guards", "passed"}})
                     .dump()
              << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

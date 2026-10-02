// SPDX-License-Identifier: Apache-2.0
#include "rwkv/inference/model.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rwkv::inference {
State Model::initial_state() const {
  State s;
  for (size_t i = 0; i < w_.layers(); ++i)
    s.layers.push_back({Vector(w_.channels()), Vector(w_.channels()),
                        Vector(w_.channels() * w_.head_size())});
  return s;
}
Vector Model::forward(int token, State &state) const {
  return evaluate({token}, state, false).front();
}
std::vector<Vector> Model::prefill(const std::vector<int> &tokens,
                                   State &state) const {
  std::vector<Vector> output;
  // Bounded staging. Each chunk uses a distinct sequence transition on NPU.
  for (size_t start = 0; start < tokens.size(); start += 16) {
    auto end = std::min(tokens.size(), start + 16);
    auto chunk =
        evaluate(std::vector<int>(tokens.begin() + start, tokens.begin() + end),
                 state, true);
    for (auto &row : chunk)
      output.push_back(std::move(row));
  }
  return output;
}
std::vector<Vector> Model::evaluate(const std::vector<int> &tokens,
                                    State &state, bool sequence) const {
  if (tokens.empty())
    return {};
  const size_t c = w_.channels(), n = w_.head_size(), t = tokens.size();
  for (int token : tokens)
    if (token < 0 || size_t(token) >= w_.vocabulary())
      throw std::runtime_error("Token outside model vocabulary");
  if (state.layers.size() != w_.layers())
    throw std::runtime_error("State layer count mismatch");
  for (const auto &s : state.layers)
    if (s.attention_shift.size() != c || s.ffn_shift.size() != c ||
        s.matrix.size() != c * n)
      throw std::runtime_error("State dimensions mismatch");
  const auto &emb = w_.at("emb.weight").data;
  std::vector<Vector> x(t), first_v(t);
  for (size_t time = 0; time < t; ++time) {
    size_t token = size_t(tokens[time]);
    x[time] = Vector(emb.begin() + token * c, emb.begin() + (token + 1) * c);
    x[time] = backend_.norm(x[time], w_.at("blocks.0.ln0.weight"),
                            w_.at("blocks.0.ln0.bias"), c, 1e-5f);
  }
  for (size_t l = 0; l < w_.layers(); ++l) {
    std::string p = "blocks." + std::to_string(l) + ".";
    auto get = [&](const std::string &name) -> const Tensor & {
      return w_.at(p + name);
    };
    auto &s = state.layers[l];
    std::vector<Vector> r(t), k(t), v(t), decay(t), neg(t), b(t), g(t);
    for (size_t time = 0; time < t; ++time) {
      auto xx =
          backend_.norm(x[time], get("ln1.weight"), get("ln1.bias"), c, 1e-5f);
      auto mix = [&](const std::string &name) {
        return backend_.element(Op::Mix, xx, s.attention_shift,
                                get("att.x_" + name).data);
      };
      auto xr = mix("r"), xw = mix("w"), xk = mix("k"), xv = mix("v"),
           xa = mix("a"), xg = mix("g");
      s.attention_shift = xx;
      auto rank = [&](const Vector &in, const std::string &name,
                      int activation) {
        auto z = backend_.linear(in, get("att." + name + "1"), true);
        if (activation == 1)
          z = backend_.element(Op::Tanh, z);
        if (activation == 2)
          z = backend_.element(Op::Sigmoid, z);
        return backend_.linear(z, get("att." + name + "2"), true);
      };
      r[time] = backend_.linear(xr, get("att.receptance.weight"));
      k[time] = backend_.linear(xk, get("att.key.weight"));
      v[time] = backend_.linear(xv, get("att.value.weight"));
      decay[time] = rank(xw, "w", 1);
      auto alpha = rank(xa, "a", 0);
      g[time] = rank(xg, "g", 2);
      if (l == 0)
        first_v[time] = v[time];
      else
        v[time] = backend_.element(Op::ValueResidual, v[time], first_v[time],
                                   rank(xv, "v", 0), get("att.v0").data);
      auto kk = backend_.element(Op::NormalizeKey, k[time], get("att.k_k").data,
                                 {}, {}, n);
      alpha = backend_.element(Op::Sigmoid, alpha, get("att.a0").data);
      neg[time] = backend_.element(Op::Negate, kk);
      b[time] = backend_.element(Op::Multiply, kk, alpha);
      k[time] =
          backend_.element(Op::KeyScale, k[time], alpha, get("att.k_a").data);
      decay[time] =
          backend_.element(Op::Decay, decay[time], get("att.w0").data);
    }
    std::vector<Vector> y;
    if (sequence)
      y = backend_.prefill(s.matrix, r, decay, k, v, neg, b, n);
    else
      y = {
          backend_.step(s.matrix, r[0], decay[0], k[0], v[0], neg[0], b[0], n)};
    for (size_t time = 0; time < t; ++time) {
      auto z = backend_.norm(y[time], get("att.ln_x.weight"),
                             get("att.ln_x.bias"), n, 64e-5f);
      auto residual = backend_.element(Op::Rkv, r[time], k[time],
                                       get("att.r_k").data, v[time], n);
      z = backend_.element(Op::Multiply, backend_.element(Op::Add, z, residual),
                           g[time]);
      x[time] = backend_.element(Op::Add, x[time],
                                 backend_.linear(z, get("att.output.weight")));
      auto xx =
          backend_.norm(x[time], get("ln2.weight"), get("ln2.bias"), c, 1e-5f);
      auto mixed =
          backend_.element(Op::Mix, xx, s.ffn_shift, get("ffn.x_k").data);
      s.ffn_shift = xx;
      auto f = backend_.element(Op::ReluSquared,
                                backend_.linear(mixed, get("ffn.key.weight")));
      x[time] = backend_.element(Op::Add, x[time],
                                 backend_.linear(f, get("ffn.value.weight")));
    }
  }
  for (auto &row : x) {
    row = backend_.linear(backend_.norm(row, w_.at("ln_out.weight"),
                                        w_.at("ln_out.bias"), c, 1e-5f),
                          w_.at("head.weight"));
    for (float f : row)
      if (!std::isfinite(f))
        throw std::runtime_error("Nonfinite logits; discard this state");
  }
  return x;
}
} // namespace rwkv::inference

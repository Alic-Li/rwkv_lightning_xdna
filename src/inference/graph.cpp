// SPDX-License-Identifier: Apache-2.0
#include "graph_internal.hpp"
#include <algorithm>
#include <stdexcept>

namespace rwkv::inference {
DecodeGraph::Impl::Id DecodeGraph::Impl::allocate(size_t size) {
  Id id = buffers.size();
  buffers.push_back({size, nullptr, Vector(size)});
  return id;
}
DecodeGraph::Impl::Id DecodeGraph::Impl::constant(const Vector &v) {
  auto it = constants.find(&v);
  if (it != constants.end())
    return it->second;
  Id id = buffers.size();
  buffers.push_back({v.size(), &v, {}});
  constants[&v] = id;
  return id;
}
const Vector &DecodeGraph::Impl::read(Id id) const {
  return id == none                ? empty
         : buffers.at(id).constant ? *buffers[id].constant
                                   : buffers[id].value;
}
DecodeGraph::Impl::Id DecodeGraph::Impl::element(Op op, Id x, Id y, Id z, Id w,
                                                 size_t group, float eps) {
  Node node;
  node.layer = recording_layer;
  node.kind = Kind::Element;
  node.op = op;
  node.inputs = {x, y, z, w, none, none, none};
  node.group = group;
  node.epsilon = eps;
  node.output = allocate(buffers.at(x).size);
  nodes.push_back(node);
  return node.output;
}
DecodeGraph::Impl::Id DecodeGraph::Impl::norm(Id x, const Tensor &w,
                                              const Tensor &b, size_t group,
                                              float eps) {
  return element(Op::Norm, x, constant(w.data), constant(b.data), none, group,
                 eps);
}
DecodeGraph::Impl::Id DecodeGraph::Impl::linear(Id x, const Tensor &w,
                                                bool transpose) {
  size_t input = transpose ? w.shape.at(0) : w.shape.at(1),
         output = transpose ? w.shape.at(1) : w.shape.at(0);
  if (buffers.at(x).size != input)
    throw std::runtime_error("Graph projection shape mismatch");
  Node node;
  node.layer = recording_layer;
  node.kind = Kind::Linear;
  node.inputs[0] = x;
  node.weight = &w;
  node.transpose = transpose;
  node.output = allocate(output);
  nodes.push_back(node);
  return node.output;
}
DecodeGraph::Impl::Id DecodeGraph::Impl::recurrent(Id state, Id r, Id d, Id k,
                                                   Id v, Id a, Id b) {
  Node node;
  node.layer = recording_layer;
  node.kind = Kind::Recurrent;
  node.inputs = {state, r, d, k, v, a, b};
  node.group = weights.head_size();
  node.output = allocate(weights.channels());
  nodes.push_back(node);
  return node.output;
}
DecodeGraph::Impl::Impl(const Weights &w, RecurrentBackend *b,
                        const std::filesystem::path &resident, WeightMode mode,
                        PrefillMode prefill)
    : weights(w), backend(b), weight_mode(mode),
      capture_prefill(prefill != PrefillMode::Sequential),
      prefill_chunk_tokens(prefill == PrefillMode::Chunked4 ? 4 : 2) {
  if (prefill != PrefillMode::Sequential && prefill != PrefillMode::Batched2 && prefill != PrefillMode::Chunked4)
    throw std::invalid_argument("Invalid prefill mode");
  if (capture_prefill && resident.empty())
    throw std::invalid_argument("Batched prefill requires NPU artifacts");
  if (mode != WeightMode::BFloat16 && mode != WeightMode::Int8FFN &&
      mode != WeightMode::Int8FFNOutput)
    throw std::invalid_argument("Invalid weight mode");
  const size_t c = w.channels(), n = w.head_size();
  embedding = allocate(c);
  Id x = norm(embedding, w.at("blocks.0.ln0.weight"), w.at("blocks.0.ln0.bias"),
              c, 1e-5f),
     first_v = none;
  for (size_t layer = 0; layer < w.layers(); ++layer) {
    recording_layer = static_cast<int>(layer);
    std::string prefix = "blocks." + std::to_string(layer) + ".";
    auto get = [&](const std::string &key) -> const Tensor & {
      return w.at(prefix + key);
    };
    auto coeff = [&](const std::string &key) {
      return constant(get(key).data);
    };
    Id old_att = allocate(c), old_ffn = allocate(c), matrix = allocate(c * n);
    Id xx = norm(x, get("ln1.weight"), get("ln1.bias"), c, 1e-5f);
    auto mix = [&](const std::string &name) {
      return element(Op::Mix, xx, old_att, coeff("att.x_" + name));
    };
    Id xr = mix("r"), xw = mix("w"), xk = mix("k"), xv = mix("v"),
       xa = mix("a"), xg = mix("g");
    auto rank = [&](Id in, const std::string &name, int activation) {
      Id z = linear(in, get("att." + name + "1"), true);
      if (activation == 1)
        z = element(Op::Tanh, z);
      if (activation == 2)
        z = element(Op::Sigmoid, z);
      return linear(z, get("att." + name + "2"), true);
    };
    Id r = linear(xr, get("att.receptance.weight")),
       k = linear(xk, get("att.key.weight")),
       v = linear(xv, get("att.value.weight"));
    Id d = rank(xw, "w", 1), a = rank(xa, "a", 0), g = rank(xg, "g", 2);
    if (layer == 0)
      first_v = v;
    else
      v = element(Op::ValueResidual, v, first_v, rank(xv, "v", 0),
                  coeff("att.v0"));
    Id kk = element(Op::NormalizeKey, k, coeff("att.k_k"), none, none, n);
    a = element(Op::Sigmoid, a, coeff("att.a0"));
    Id neg = element(Op::Negate, kk), kb = element(Op::Multiply, kk, a);
    k = element(Op::KeyScale, k, a, coeff("att.k_a"));
    d = element(Op::Decay, d, coeff("att.w0"));
    Id y = recurrent(matrix, r, d, k, v, neg, kb);
    y = norm(y, get("att.ln_x.weight"), get("att.ln_x.bias"), n, 64e-5f);
    Id residual = element(Op::Rkv, r, k, coeff("att.r_k"), v, n);
    y = element(Op::Multiply, element(Op::Add, y, residual), g);
    x = element(Op::Add, x, linear(y, get("att.output.weight")));
    Id ff = norm(x, get("ln2.weight"), get("ln2.bias"), c, 1e-5f);
    Id f = linear(element(Op::Mix, ff, old_ffn, coeff("ffn.x_k")),
                  get("ffn.key.weight"));
    f = element(Op::ReluSquared, f);
    x = element(Op::Add, x, linear(f, get("ffn.value.weight")));
    states.push_back({old_att, old_ffn, matrix, xx, ff});
  }
  recording_layer = -1;
  logits = linear(norm(x, w.at("ln_out.weight"), w.at("ln_out.bias"), c, 1e-5f),
                  w.at("head.weight"));
  // Check the recorded dependency schedule before any replay is allowed.
  std::vector<bool> ready(buffers.size(), true);
  for (const auto &node : nodes)
    ready[node.output] = false;
  for (const auto &node : nodes) {
    for (Id id : node.inputs)
      if (id != none && !ready.at(id))
        throw std::runtime_error("Non-topological graph dependency");
    ready[node.output] = true;
  }
  if (!resident.empty())
    prepare_resident(resident);
}
DecodeGraph::DecodeGraph(const Weights &w, RecurrentBackend &b)
    : impl_(std::make_unique<Impl>(w, &b, std::filesystem::path{})) {}
DecodeGraph::DecodeGraph(const Weights &w,
                         const std::filesystem::path &resident, WeightMode mode,
                         PrefillMode prefill)
    : impl_(std::make_unique<Impl>(w, nullptr, resident, mode, prefill)) {
  if (resident.empty())
    throw std::invalid_argument("NPU artifact directory is required");
}
DecodeGraph::~DecodeGraph() = default;
Vector DecodeGraph::replay(int token, State &state) {
  return impl_->replay(token, state);
}
void DecodeGraph::load_state(const State &state) { impl_->load_state(state); }
State DecodeGraph::export_state() const { return impl_->export_state(); }
Vector DecodeGraph::replay_resident(int token) {
  State unused;
  return impl_->replay(token, unused, true);
}
Vector DecodeGraph::prefill_resident(const std::vector<int> &tokens) {
  return impl_->prefill(tokens);
}
void DecodeGraph::set_trace(Trace trace) { impl_->trace = std::move(trace); }
void DecodeGraph::set_projection_trace(ProjectionTrace trace) {
  impl_->projection_trace = std::move(trace);
}
GraphStats DecodeGraph::stats() const {
  return {impl_->nodes.size(),
          impl_->buffers.size(),
          impl_->replays,
          impl_->decode_run_count(),
          impl_->resident_bytes,
          impl_->upload_bytes,
          impl_->download_bytes,
          impl_->root_bos,
          impl_->decode_run_count(),
          impl_->weights.channels() * 4,
          impl_->weights.vocabulary() * 4,
          impl_->prefill_chunk_tokens == 2 ? impl_->prefill_body.size() + impl_->prefill_head.size() : 0,
          impl_->prefill_run_count,
          impl_->prefill_body.empty() ? 0 : impl_->prefill_chunk_tokens,
          impl_->prefill_body.size() + impl_->prefill_head.size()};
}
} // namespace rwkv::inference

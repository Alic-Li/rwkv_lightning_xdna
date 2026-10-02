// SPDX-License-Identifier: Apache-2.0
#include "rwkv/inference/graph.hpp"
#include "rwkv/xdna/session.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace rwkv::inference {
struct DecodeGraph::Impl {
  using Id = size_t;
  static constexpr Id none = std::numeric_limits<Id>::max();
  enum class Kind { Element, Linear, Recurrent };
  struct Buffer {
    size_t size;
    const Vector *constant = nullptr;
    Vector value;
  };
  struct Node {
    Kind kind;
    Op op = Op::Add;
    std::array<Id, 7> inputs{none, none, none, none, none, none, none};
    Id output = none;
    const Tensor *weight = nullptr;
    bool transpose = false;
    size_t group = 1;
    float epsilon = 0;
  };
  struct StateBinding {
    Id old_attention, old_ffn, matrix, new_attention, new_ffn;
  };
  const Weights &weights;
  RecurrentBackend &backend;
  std::vector<Buffer> buffers;
  std::vector<Node> nodes;
  std::vector<StateBinding> states;
  std::map<const Vector *, Id> constants;
  Id embedding, logits;
  size_t replays = 0;
  bool device_failed = false;
  const Vector empty;
  std::map<std::string, std::unique_ptr<xdna::Session>> sessions;
  std::vector<xdna::DeviceBuffer> device_buffers;
  std::vector<xdna::DeviceRun> runs;
  size_t resident_bytes = 0, upload_bytes = 0, download_bytes = 0;
  xdna::Session &session(const std::filesystem::path &root,
                         const std::string &name) {
    auto &s = sessions[name];
    if (!s)
      s = std::make_unique<xdna::Session>(root / name / "design.xclbin",
                                          root / name / "instructions.bin");
    return *s;
  }
  xdna::DeviceBuffer initialized(xdna::Session &s, const Vector &v) {
    auto b = s.allocate(v.size() * sizeof(float));
    b.upload(v.data(), v.size() * sizeof(float));
    resident_bytes += b.size();
    return b;
  }
  void prepare_resident(const std::filesystem::path &root) {
    for (const auto &name : {"resident-ops", "resident-decode"}) {
      std::ifstream in(root / name / "config.json");
      nlohmann::json j;
      if (!in || !(in >> j) || j.at("schema_version") != 1 ||
          j.at("abi") != name || j.at("dtype") != "float32")
        throw std::runtime_error("Missing/incompatible resident ABI: " +
                                 std::string(name));
    }
    if (weights.head_size() != 64 || weights.channels() > 2048 ||
        2048 % weights.channels())
      throw std::runtime_error(
          "Resident graph requires head_size=64 and channels dividing 2048");
    auto &ops = session(root, "resident-ops");
    auto &wkv = session(root, "resident-decode");
    device_buffers.resize(buffers.size());
    std::map<Id, xdna::DeviceBuffer> recurrent_vectors;
    for (const auto &node : nodes) {
      if (node.kind != Kind::Recurrent)
        continue;
      auto packed = initialized(ops, Vector(6 * 2048, 0));
      recurrent_vectors.emplace(node.output, packed);
      for (size_t i = 1; i < 7; ++i) {
        Id id = node.inputs[i];
        if (device_buffers[id].size() || buffers[id].size > 2048)
          throw std::runtime_error("Unsupported recurrent vector alias");
        device_buffers[id] = packed.slice((i - 1) * 2048 * 4, 2048 * 4);
      }
    }
    for (size_t id = 0; id < buffers.size(); ++id) {
      if (device_buffers[id].size())
        continue;
      const auto &b = buffers[id];
      Vector value(((b.size + 2047) / 2048) * 2048, 0);
      if (b.constant)
        std::copy(b.constant->begin(), b.constant->end(), value.begin());
      device_buffers[id] = initialized(ops, value);
    }
    auto zero = initialized(ops, Vector(2048, 0));
    auto slice = [&](Id id, size_t start, size_t count) {
      return device_buffers.at(id).slice(start * 4, count * 4);
    };
    for (const auto &node : nodes) {
      const auto &a = node.inputs;
      if (node.kind == Kind::Element) {
        for (size_t start = 0; start < buffers[node.output].size;
             start += 2048) {
          Vector meta(16, 0);
          meta[0] = static_cast<int>(node.op);
          meta[1] = std::min(size_t(2048), buffers[node.output].size - start);
          meta[2] = node.group;
          meta[3] = node.epsilon;
          std::vector<xdna::DeviceBuffer> args{initialized(ops, meta)};
          for (size_t i = 0; i < 4; ++i)
            args.push_back(a[i] == none ? zero : slice(a[i], start, 2048));
          args.push_back(slice(node.output, start, 2048));
          runs.push_back(ops.prepare(args));
        }
      } else if (node.kind == Kind::Linear) {
        const auto &w = *node.weight;
        size_t inputs = buffers[a[0]].size, outputs = buffers[node.output].size;
        size_t k = ((inputs + 255) / 256) * 256;
        auto &gemv = session(root, "gemv-" + std::to_string(k));
        for (size_t start = 0; start < outputs; start += 256) {
          Vector packed(256 * k, 0);
          for (size_t r = 0; r < std::min(size_t(256), outputs - start); ++r)
            for (size_t col = 0; col < inputs; ++col) {
              size_t pos = ((r / 16) * (k / 256) + col / 256) * 4096 +
                           (r % 16) * 256 + col % 256;
              packed[pos] = w.data[node.transpose ? col * outputs + start + r
                                                  : (start + r) * inputs + col];
            }
          auto weight = initialized(gemv, packed);
          runs.push_back(gemv.prepare(
              {slice(a[0], 0, k), weight, slice(node.output, start, 256)}));
        }
      } else {
        for (size_t h = 0; h < weights.heads(); ++h) {
          std::vector<xdna::DeviceBuffer> args{
              slice(a[0], h * 4096, 4096),
              recurrent_vectors.at(node.output).slice(h * 64 * 4, 10304 * 4),
              slice(node.output, h * 64, 64)};
          runs.push_back(wkv.prepare(args));
        }
      }
    }
    upload_bytes = weights.channels() * 4;
    download_bytes = weights.vocabulary() * 4;
    for (const auto &s : states) {
      size_t bytes = (buffers[s.old_attention].size + buffers[s.old_ffn].size +
                      buffers[s.matrix].size) *
                     4;
      upload_bytes += bytes;
      download_bytes += bytes;
    }
  }
  Id allocate(size_t size) {
    Id id = buffers.size();
    buffers.push_back({size, nullptr, Vector(size)});
    return id;
  }
  Id constant(const Vector &v) {
    auto it = constants.find(&v);
    if (it != constants.end())
      return it->second;
    Id id = buffers.size();
    buffers.push_back({v.size(), &v, {}});
    constants[&v] = id;
    return id;
  }
  const Vector &read(Id id) const {
    return id == none                ? empty
           : buffers.at(id).constant ? *buffers[id].constant
                                     : buffers[id].value;
  }
  Id element(Op op, Id x, Id y = none, Id z = none, Id w = none,
             size_t group = 1, float eps = 0) {
    Node node;
    node.kind = Kind::Element;
    node.op = op;
    node.inputs = {x, y, z, w, none, none, none};
    node.group = group;
    node.epsilon = eps;
    node.output = allocate(buffers.at(x).size);
    nodes.push_back(node);
    return node.output;
  }
  Id norm(Id x, const Tensor &w, const Tensor &b, size_t group, float eps) {
    return element(Op::Norm, x, constant(w.data), constant(b.data), none, group,
                   eps);
  }
  Id linear(Id x, const Tensor &w, bool transpose = false) {
    size_t input = transpose ? w.shape.at(0) : w.shape.at(1),
           output = transpose ? w.shape.at(1) : w.shape.at(0);
    if (buffers.at(x).size != input)
      throw std::runtime_error("Graph projection shape mismatch");
    Node node;
    node.kind = Kind::Linear;
    node.inputs[0] = x;
    node.weight = &w;
    node.transpose = transpose;
    node.output = allocate(output);
    nodes.push_back(node);
    return node.output;
  }
  Id recurrent(Id state, Id r, Id d, Id k, Id v, Id a, Id b) {
    Node node;
    node.kind = Kind::Recurrent;
    node.inputs = {state, r, d, k, v, a, b};
    node.group = weights.head_size();
    node.output = allocate(weights.channels());
    nodes.push_back(node);
    return node.output;
  }
  Impl(const Weights &w, RecurrentBackend &b,
       const std::filesystem::path &resident)
      : weights(w), backend(b) {
    const size_t c = w.channels(), n = w.head_size();
    embedding = allocate(c);
    Id x = norm(embedding, w.at("blocks.0.ln0.weight"),
                w.at("blocks.0.ln0.bias"), c, 1e-5f),
       first_v = none;
    for (size_t layer = 0; layer < w.layers(); ++layer) {
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
    logits =
        linear(norm(x, w.at("ln_out.weight"), w.at("ln_out.bias"), c, 1e-5f),
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
  Vector replay(int token, State &state) {
    if (device_failed)
      throw std::runtime_error(
          "Recreate resident graph after a device dispatch failure");
    const size_t c = weights.channels(), n = weights.head_size();
    if (token < 0 || size_t(token) >= weights.vocabulary() ||
        state.layers.size() != states.size())
      throw std::runtime_error("Invalid graph token/state");
    auto bind = [&](Id id, const Vector &v) {
      if (v.size() != buffers[id].size)
        throw std::runtime_error("Graph state dimension mismatch");
      std::copy(v.begin(), v.end(), buffers[id].value.begin());
    };
    for (size_t i = 0; i < states.size(); ++i) {
      auto &s = state.layers[i];
      if (s.attention_shift.size() != c || s.ffn_shift.size() != c ||
          s.matrix.size() != c * n)
        throw std::runtime_error("Graph state dimensions mismatch");
    }
    const auto &emb = weights.at("emb.weight").data;
    std::copy_n(emb.data() + size_t(token) * c, c,
                buffers[embedding].value.data());
    for (size_t i = 0; i < states.size(); ++i) {
      bind(states[i].old_attention, state.layers[i].attention_shift);
      bind(states[i].old_ffn, state.layers[i].ffn_shift);
      bind(states[i].matrix, state.layers[i].matrix);
    }
    if (!device_buffers.empty()) {
      auto upload = [&](Id id) {
        const auto &v = read(id);
        device_buffers[id].upload(v.data(), v.size() * 4);
      };
      upload(embedding);
      for (const auto &s : states) {
        upload(s.old_attention);
        upload(s.old_ffn);
        upload(s.matrix);
      }
      for (size_t i = 0; i < runs.size(); ++i) {
        try {
          runs[i].execute();
        } catch (const std::exception &e) {
          device_failed = true;
          throw std::runtime_error("Resident run " + std::to_string(i) + ": " +
                                   e.what());
        }
      }
      auto download = [&](Id id) {
        auto &v = buffers[id].value;
        device_buffers[id].download(v.data(), v.size() * 4);
      };
      download(logits);
      for (const auto &s : states) {
        download(s.new_attention);
        download(s.new_ffn);
        download(s.matrix);
      }
    } else
      for (const auto &node : nodes) {
        const auto &a = node.inputs;
        Vector result;
        if (node.kind == Kind::Element)
          result = backend.element(node.op, read(a[0]), read(a[1]), read(a[2]),
                                   read(a[3]), node.group, node.epsilon);
        else if (node.kind == Kind::Linear)
          result = backend.linear(read(a[0]), *node.weight, node.transpose);
        else
          result = backend.step(buffers[a[0]].value, read(a[1]), read(a[2]),
                                read(a[3]), read(a[4]), read(a[5]), read(a[6]),
                                node.group);
        if (result.size() != buffers[node.output].size)
          throw std::runtime_error("Graph output shape mismatch");
        std::copy(result.begin(), result.end(),
                  buffers[node.output].value.begin());
      }
    for (float f : read(logits))
      if (!std::isfinite(f))
        throw std::runtime_error("Nonfinite graph logits");
    // Commit only after the complete token succeeds. No recorded inputs are
    // replayed.
    for (size_t i = 0; i < states.size(); ++i) {
      state.layers[i].attention_shift = read(states[i].new_attention);
      state.layers[i].ffn_shift = read(states[i].new_ffn);
      state.layers[i].matrix = read(states[i].matrix);
    }
    ++replays;
    return read(logits);
  }
};
DecodeGraph::DecodeGraph(const Weights &w, RecurrentBackend &b,
                         const std::filesystem::path &resident)
    : impl_(std::make_unique<Impl>(w, b, resident)) {}
DecodeGraph::~DecodeGraph() = default;
Vector DecodeGraph::replay(int token, State &state) {
  return impl_->replay(token, state);
}
GraphStats DecodeGraph::stats() const {
  return {impl_->nodes.size(),  impl_->buffers.size(), impl_->replays,
          impl_->runs.size(),   impl_->resident_bytes, impl_->upload_bytes,
          impl_->download_bytes};
}
} // namespace rwkv::inference

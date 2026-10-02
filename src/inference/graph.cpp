// SPDX-License-Identifier: Apache-2.0
#include "rwkv/inference/graph.hpp"
#include "rwkv/xdna/session.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
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
  DecodeGraph::Trace trace;
  std::vector<size_t> node_run_ends;
  bool device_failed = false, resident_state_valid = false;
  std::vector<xdna::DeviceRun> shift_runs;
  const Vector empty;
  std::map<std::string, std::unique_ptr<xdna::Session>> sessions;
  std::vector<xdna::DeviceBuffer> device_buffers;
  std::vector<xdna::DeviceRun> runs;
  size_t resident_bytes = 0, upload_bytes = 0, download_bytes = 0, root_bos = 0;
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
    ++root_bos;
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
    const std::map<std::string, nlohmann::json> optional_abis = {
        {"fused-value",
         {{"schema_version", 1}, {"dtype", "float32"}, {"channels", 2048}}},
        {"upstream-norm",
         {{"schema_version", 1},
          {"dtype", "float32"},
          {"channels", 2048},
          {"epsilon", 1e-5}}},
        {"fused-mix",
         {{"schema_version", 3},
          {"dtype", "float32"},
          {"count", 2048},
          {"mixes", 6},
          {"state_update", true}}},
        {"fused-shift-mix",
         {{"schema_version", 1},
          {"dtype", "float32"},
          {"count", 2048},
          {"state_update", true}}},
        {"fused-prepare",
         {{"schema_version", 2},
          {"dtype", "float32"},
          {"channels", 2048},
          {"head_size", 64},
          {"stage", "prepare"}}},
        {"fused-finish",
         {{"schema_version", 2},
          {"dtype", "float32"},
          {"channels", 2048},
          {"head_size", 64},
          {"stage", "finish"}}},
        {"fused-ffn-key",
         {{"schema_version", 1},
          {"dtype", "float32"},
          {"rows", 8192},
          {"k", 2048}}},
        {"resident-ops-fast",
         {{"schema_version", 1},
          {"dtype", "float32"},
          {"abi", "resident-ops-fast"}}}};
    auto check_optional = [&](const std::string &name,
                              const nlohmann::json &expected) {
      if (!std::filesystem::exists(root / name / "config.json"))
        return;
      std::ifstream config(root / name / "config.json");
      nlohmann::json j;
      config >> j;
      for (auto it = expected.begin(); it != expected.end(); ++it)
        if (!j.contains(it.key()) || j.at(it.key()) != it.value())
          throw std::runtime_error("Incompatible fused ABI: " + name);
    };
    for (const auto &entry : optional_abis)
      check_optional(entry.first, entry.second);
    for (int a = 0; a < 3; ++a)
      check_optional("fused-rank-" + std::to_string(a), {{"schema_version", 1},
                                                         {"dtype", "float32"},
                                                         {"channels", 2048},
                                                         {"rank", 256},
                                                         {"activation", a}});
    if (weights.head_size() != 64 || weights.channels() > 2048 ||
        2048 % weights.channels())
      throw std::runtime_error(
          "Resident graph requires head_size=64 and channels dividing 2048");
    bool exact_fp32 = false;
    if (std::filesystem::exists(root / "resident-ops-fast/config.json")) {
      std::ifstream config(root / "resident-ops-fast/config.json");
      nlohmann::json j;
      config >> j;
      exact_fp32 = j.value("exact_fp32", false);
    }
    for (const auto &name : {"array-decode", "fused-mix", "fused-shift-mix",
                             "fused-rank-0", "fused-rank-1", "fused-rank-2",
                             "fused-prepare", "fused-finish", "fused-value"}) {
      if (!std::filesystem::exists(root / name / "config.json"))
        continue;
      std::ifstream config(root / name / "config.json");
      nlohmann::json j;
      config >> j;
      if (!j.contains("exact_fp32") || j.at("exact_fp32") != exact_fp32)
        throw std::runtime_error(
            "Mixed/missing FP32 arithmetic contract: " + std::string(name) +
            "; rebuild with rwkv7_optimized.py");
    }
    auto &ops = session(
        root, std::filesystem::exists(root / "resident-ops-fast/config.json")
                  ? "resident-ops-fast"
                  : "resident-ops");
    const bool array_wkv =
        weights.heads() == 32 &&
        std::filesystem::exists(root / "array-decode/config.json");
    if (array_wkv) {
      std::ifstream config(root / "array-decode/config.json");
      nlohmann::json j;
      config >> j;
      if (j.at("schema_version") != 1 || j.at("dtype") != "float32" ||
          j.at("heads") != 32 || j.at("head_size") != 64)
        throw std::runtime_error("Incompatible array WKV ABI");
    }
    auto &wkv = session(root, array_wkv ? "array-decode" : "resident-decode");
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
    std::map<size_t, std::vector<xdna::DeviceBuffer>> stage_args;
    auto pack_vectors = [&](const std::vector<Id> &ids, size_t slots) {
      Vector value(slots * 2048, 0);
      for (size_t i = 0; i < ids.size(); ++i) {
        if (buffers[ids[i]].constant)
          std::copy(read(ids[i]).begin(), read(ids[i]).end(),
                    value.begin() + i * 2048);
        if (device_buffers[ids[i]].size())
          throw std::runtime_error("Stage input alias");
      }
      auto packed = initialized(ops, value);
      for (size_t i = 0; i < ids.size(); ++i)
        device_buffers[ids[i]] = packed.slice(i * 2048 * 4, 2048 * 4);
      return packed;
    };
    for (size_t i = 0; i < nodes.size(); ++i) {
      const auto *n = &nodes[i];
      if (n->kind != Kind::Element)
        continue;
      if (n->op == Op::ValueResidual &&
          std::filesystem::exists(root / "fused-value/config.json")) {
        auto input =
            pack_vectors({n->inputs[0], n->inputs[2], n->inputs[3]}, 3);
        stage_args[i] = {input, device_buffers[n->inputs[1]],
                         device_buffers[n->output]};
      }

      if (n->op == Op::NormalizeKey &&
          std::filesystem::exists(root / "fused-prepare/config.json")) {
        if (i + 6 >= nodes.size() || n[6].kind != Kind::Recurrent)
          throw std::runtime_error("Stage topology");
        auto input = pack_vectors(
            {n[0].inputs[0], n[1].inputs[0], n[5].inputs[0], n[0].inputs[1],
             n[4].inputs[2], n[1].inputs[1], n[5].inputs[1]},
            8);
        auto out = pack_vectors({n[0].output, n[1].output}, 2);
        stage_args[i] = {input, recurrent_vectors.at(n[6].output), out};
      }
      if (n->op == Op::Norm && n->group == 64 && i &&
          nodes[i - 1].kind == Kind::Recurrent &&
          std::filesystem::exists(root / "fused-finish/config.json")) {
        auto input =
            pack_vectors({n[0].inputs[0], n[0].inputs[1], n[0].inputs[2],
                          n[1].inputs[2], n[3].inputs[1]},
                         5);
        auto out = pack_vectors(
            {n[0].output, n[1].output, n[2].output, n[3].output}, 6);
        stage_args[i] = {input, recurrent_vectors.at(nodes[i - 1].output), out};
      }
    }
    // One root BO for constants and activation slots. Offsets stay fixed;
    // recurrent gather slots have their own six-vector backing allocation.
    size_t arena_floats = 0;
    for (size_t id = 0; id < buffers.size(); ++id)
      if (!device_buffers[id].size())
        arena_floats += ((buffers[id].size + 2047) / 2048) * 2048;
    Vector arena_data(arena_floats, 0);
    size_t offset = 0;
    for (size_t id = 0; id < buffers.size(); ++id) {
      if (device_buffers[id].size())
        continue;
      const auto &b = buffers[id];
      if (b.constant)
        std::copy(b.constant->begin(), b.constant->end(),
                  arena_data.begin() + offset);
      offset += ((b.size + 2047) / 2048) * 2048;
    }
    auto arena = initialized(ops, arena_data);
    offset = 0;
    for (size_t id = 0; id < buffers.size(); ++id) {
      if (device_buffers[id].size())
        continue;
      size_t size = ((buffers[id].size + 2047) / 2048) * 2048;
      device_buffers[id] = arena.slice(offset * 4, size * 4);
      offset += size;
    }
    auto zero = initialized(ops, Vector(2048, 0));
    std::map<std::array<float, 4>, xdna::DeviceBuffer> metadata;
    auto meta_buffer = [&](const Vector &v) {
      const std::array<float, 4> key{v[0], v[1], v[2], v[3]};
      auto &b = metadata[key];
      if (!b.size())
        b = initialized(ops, v);
      return b;
    };
    auto slice = [&](Id id, size_t start, size_t count) {
      return device_buffers.at(id).slice(start * 4, count * 4);
    };
    for (size_t node_index = 0; node_index < nodes.size(); ++node_index) {
      const auto &node = nodes[node_index];
      const auto &a = node.inputs;
      if (node.kind == Kind::Linear && node.transpose &&
          buffers[a[0]].size == 2048 && buffers[node.output].size <= 256) {
        size_t last = node_index + 1;
        int activation = 0;
        if (last < nodes.size() && nodes[last].kind == Kind::Element &&
            (nodes[last].op == Op::Tanh || nodes[last].op == Op::Sigmoid) &&
            nodes[last].inputs[0] == node.output &&
            nodes[last].inputs[1] == none) {
          activation = nodes[last].op == Op::Tanh ? 1 : 2;
          ++last;
        }
        const auto name = "fused-rank-" + std::to_string(activation);
        if (last < nodes.size() && nodes[last].kind == Kind::Linear &&
            nodes[last].transpose &&
            nodes[last].inputs[0] == nodes[last - 1].output &&
            buffers[nodes[last].output].size == 2048 &&
            std::filesystem::exists(root / name / "config.json")) {
          auto &rank = session(root, name);
          auto pack = [&](const Tensor &w, size_t in, size_t out,
                          size_t padded_in, size_t padded_out) {
            Vector v(padded_in * padded_out, 0);
            for (size_t row = 0; row < out; ++row)
              for (size_t col = 0; col < in; ++col) {
                size_t pos =
                    ((row / 16) * (padded_in / 256) + col / 256) * 4096 +
                    (row % 16) * 256 + col % 256;
                v[pos] = w.data[col * out + row];
              }
            return initialized(rank, v);
          };
          const size_t hidden = buffers[node.output].size;
          auto w1 = pack(*node.weight, 2048, hidden, 2048, 256);
          auto w2 = pack(*nodes[last].weight, hidden, 2048, 256, 2048);
          auto activated = activation ? device_buffers[nodes[last - 1].output]
                                      : initialized(rank, Vector(256, 0));
          runs.push_back(rank.prepare({device_buffers[a[0]], w1, w2,
                                       device_buffers[node.output], activated,
                                       device_buffers[nodes[last].output]}));
          for (size_t j = node_index; j <= last; ++j)
            node_run_ends.push_back(runs.size());
          node_index = last;
          continue;
        }
      }
      auto is_op = [&](size_t offset, Op op) {
        return node_index + offset < nodes.size() &&
               nodes[node_index + offset].kind == Kind::Element &&
               nodes[node_index + offset].op == op;
      };
      if (is_op(0, Op::NormalizeKey) && is_op(1, Op::Sigmoid) &&
          is_op(2, Op::Negate) && is_op(3, Op::Multiply) &&
          is_op(4, Op::KeyScale) && is_op(5, Op::Decay) &&
          std::filesystem::exists(root / "fused-prepare/config.json")) {
        const auto *n = &nodes[node_index];
        if (n[2].inputs[0] != n[0].output || n[3].inputs[0] != n[0].output ||
            n[3].inputs[1] != n[1].output || n[4].inputs[0] != n[0].inputs[0] ||
            n[4].inputs[1] != n[1].output || n[0].group != 64)
          throw std::runtime_error(
              "Unexpected attention preparation dependencies");
        auto &stage = session(root, "fused-prepare");
        runs.push_back(stage.prepare(stage_args.at(node_index)));
        for (size_t j = 0; j < 6; ++j)
          node_run_ends.push_back(runs.size());
        node_index += 5;
        continue;
      }
      if (is_op(0, Op::Norm) && node.group == 64 && is_op(1, Op::Rkv) &&
          is_op(2, Op::Add) && is_op(3, Op::Multiply) &&
          std::filesystem::exists(root / "fused-finish/config.json")) {
        const auto *n = &nodes[node_index];
        if (n[2].inputs[0] != n[0].output || n[2].inputs[1] != n[1].output ||
            n[3].inputs[0] != n[2].output || n[1].group != 64 ||
            n[0].epsilon != 64e-5f)
          throw std::runtime_error("Unexpected attention finish dependencies");
        auto &stage = session(root, "fused-finish");
        runs.push_back(stage.prepare(stage_args.at(node_index)));
        for (size_t j = 0; j < 4; ++j)
          node_run_ends.push_back(runs.size());
        node_index += 3;
        continue;
      }
      if (node.kind == Kind::Element && node.op == Op::Mix &&
          node_index + 6 <= nodes.size() &&
          std::filesystem::exists(root / "fused-mix/config.json")) {
        bool six = true;
        for (size_t j = 0; j < 6; ++j) {
          const auto &m = nodes[node_index + j];
          six &= m.kind == Kind::Element && m.op == Op::Mix &&
                 m.inputs[0] == a[0] && m.inputs[1] == a[1];
        }
        if (six) {
          auto &mix = session(root, "fused-mix");
          Vector coeff(6 * 2048, 0);
          for (size_t j = 0; j < 6; ++j) {
            const auto &v = read(nodes[node_index + j].inputs[2]);
            std::copy(v.begin(), v.end(), coeff.begin() + j * 2048);
          }
          auto mixed = initialized(mix, Vector(6 * 2048, 0));
          for (size_t j = 0; j < 6; ++j)
            device_buffers[nodes[node_index + j].output] =
                mixed.slice(j * 2048 * 4, 2048 * 4);
          std::vector<xdna::DeviceBuffer> args{device_buffers[a[0]],
                                               device_buffers[a[1]],
                                               initialized(mix, coeff), mixed};
          runs.push_back(mix.prepare(args));
          for (size_t j = 0; j < 6; ++j)
            node_run_ends.push_back(runs.size());
          node_index += 5;
          continue;
        }
      }
      if (node.kind == Kind::Element && node.op == Op::ValueResidual &&
          std::filesystem::exists(root / "fused-value/config.json")) {
        auto &value = session(root, "fused-value");
        runs.push_back(value.prepare(stage_args.at(node_index)));
      } else if (!exact_fp32 && node.kind == Kind::Element &&
                 node.op == Op::Norm && node.group == 2048 &&
                 node.epsilon == 1e-5f &&
                 std::filesystem::exists(root / "upstream-norm/config.json")) {
        auto &norm = session(root, "upstream-norm");
        runs.push_back(
            norm.prepare({device_buffers[a[0]], device_buffers[a[1]],
                          device_buffers[a[2]], device_buffers[node.output]}));
      } else if (node.kind == Kind::Element && node.op == Op::Mix &&
                 std::filesystem::exists(root /
                                         "fused-shift-mix/config.json") &&
                 std::any_of(states.begin(), states.end(),
                             [&](const StateBinding &s) {
                               return s.old_ffn == a[1];
                             })) {
        auto &mix = session(root, "fused-shift-mix");
        runs.push_back(
            mix.prepare({device_buffers[a[0]], device_buffers[a[1]],
                         device_buffers[a[2]], device_buffers[node.output]}));
      } else if (node.kind == Kind::Element) {
        for (size_t start = 0; start < buffers[node.output].size;
             start += 2048) {
          Vector meta(16, 0);
          meta[0] = static_cast<int>(node.op);
          meta[1] = std::min(size_t(2048), buffers[node.output].size - start);
          meta[2] = node.group;
          meta[3] = node.epsilon;
          std::vector<xdna::DeviceBuffer> args{meta_buffer(meta)};
          for (size_t i = 0; i < 4; ++i)
            args.push_back(a[i] == none ? zero : slice(a[i], start, 2048));
          args.push_back(slice(node.output, start, 2048));
          runs.push_back(ops.prepare(args));
        }
      } else if (node.kind == Kind::Linear) {
        const auto &w = *node.weight;
        size_t inputs = buffers[a[0]].size, outputs = buffers[node.output].size;
        size_t k = ((inputs + 255) / 256) * 256;
        auto array_name = "array-gemv-" + std::to_string(k);
        size_t array_rows = 2048;
        if (outputs >= 8192 && outputs % 8192 == 0 &&
            std::filesystem::exists(root / (array_name + "-8192") /
                                    "config.json")) {
          array_name += "-8192";
          array_rows = 8192;
        }
        auto array32_name = array_name;
        array32_name.replace(0, 5, "array32");
        const bool array32 =
            array_rows == 8192 &&
            std::filesystem::exists(root / array32_name / "config.json");
        if (array32)
          array_name = array32_name;
        const bool array =
            outputs >= 2048 &&
            std::filesystem::exists(root / array_name / "config.json");
        const size_t rows = array ? array_rows : 256;
        if (array) {
          std::ifstream config(root / array_name / "config.json");
          nlohmann::json j;
          config >> j;
          if (j.at("schema_version") != (array32 ? 2 : 1) ||
              j.at("dtype") != "float32" || j.at("rows") != rows ||
              j.at("cores") != (array32 ? 32 : 8) || j.at("k") != k ||
              (array32 && j.at("layout") != "column_block_lane"))
            throw std::runtime_error("Incompatible array GEMV ABI");
        }
        const bool ffn =
            inputs == 2048 && outputs == 8192 && rows == 8192 &&
            is_op(1, Op::ReluSquared) &&
            nodes[node_index + 1].inputs[0] == node.output &&
            std::filesystem::exists(root / "fused-ffn-key/config.json");
        auto &gemv = session(root, ffn     ? "fused-ffn-key"
                                   : array ? array_name
                                           : "gemv-" + std::to_string(k));
        for (size_t start = 0; start < outputs; start += rows) {
          Vector packed(rows * k, 0);
          for (size_t r = 0; r < std::min(rows, outputs - start); ++r)
            for (size_t col = 0; col < inputs; ++col) {
              size_t pos = ((r / 16) * (k / 256) + col / 256) * 4096 +
                           (r % 16) * 256 + col % 256;
              if (array32 && !ffn) {
                size_t stripe = rows / 32;
                pos = (((r / (stripe * 4) * (stripe / 16) + (r % stripe) / 16) *
                            (k / 256) +
                        col / 256) *
                           4 +
                       (r / stripe) % 4) *
                          4096 +
                      (r % 16) * 256 + col % 256;
              }
              packed[pos] = w.data[node.transpose ? col * outputs + start + r
                                                  : (start + r) * inputs + col];
            }
          auto weight = initialized(gemv, packed);
          std::vector<xdna::DeviceBuffer> args{slice(a[0], 0, k), weight,
                                               slice(node.output, start, rows)};
          if (ffn)
            args.push_back(device_buffers[nodes[node_index + 1].output]);
          runs.push_back(gemv.prepare(args));
        }
        if (ffn) {
          node_run_ends.push_back(runs.size());
          ++node_index;
        }
      } else if (array_wkv) {
        runs.push_back(wkv.prepare({device_buffers[a[0]],
                                    recurrent_vectors.at(node.output),
                                    device_buffers[node.output]}));
      } else {
        for (size_t h = 0; h < weights.heads(); ++h) {
          std::vector<xdna::DeviceBuffer> args{
              slice(a[0], h * 4096, 4096),
              recurrent_vectors.at(node.output).slice(h * 64 * 4, 10304 * 4),
              slice(node.output, h * 64, 64)};
          runs.push_back(wkv.prepare(args));
        }
      }
      node_run_ends.push_back(runs.size());
    }
    for (const auto &s : states) {
      Vector meta(16, 0);
      meta[0] = static_cast<int>(Op::Add);
      meta[1] = weights.channels();
      meta[2] = 1;
      for (auto pair : {std::pair<Id, Id>{s.new_attention, s.old_attention},
                        std::pair<Id, Id>{s.new_ffn, s.old_ffn}})
        if (!std::filesystem::exists(root /
                                     (pair.second == s.old_attention
                                          ? "fused-mix"
                                          : "fused-shift-mix") /
                                     "config.json"))
          shift_runs.push_back(
              ops.prepare({meta_buffer(meta), device_buffers[pair.first], zero,
                           zero, zero, device_buffers[pair.second]}));
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
  void validate_state(const State &state) const {
    if (state.layers.size() != states.size())
      throw std::runtime_error("Invalid state layers");
    for (const auto &s : state.layers)
      if (s.attention_shift.size() != weights.channels() ||
          s.ffn_shift.size() != weights.channels() ||
          s.matrix.size() != weights.channels() * weights.head_size())
        throw std::runtime_error("Invalid state dimensions");
  }
  void load_state(const State &state) {
    if (device_buffers.empty() || device_failed)
      throw std::runtime_error("Resident graph unavailable");
    validate_state(state);
    resident_state_valid = false;
    for (size_t i = 0; i < states.size(); ++i) {
      const auto &s = states[i];
      const auto &v = state.layers[i];
      device_buffers[s.old_attention].upload(v.attention_shift.data(),
                                             v.attention_shift.size() * 4);
      device_buffers[s.old_ffn].upload(v.ffn_shift.data(),
                                       v.ffn_shift.size() * 4);
      device_buffers[s.matrix].upload(v.matrix.data(), v.matrix.size() * 4);
    }
    resident_state_valid = true;
  }
  State export_state() const {
    if (!resident_state_valid || device_failed)
      throw std::runtime_error("No valid resident state");
    State out;
    for (const auto &s : states) {
      LayerState v{Vector(weights.channels()), Vector(weights.channels()),
                   Vector(weights.channels() * weights.head_size())};
      device_buffers[s.old_attention].download(v.attention_shift.data(),
                                               v.attention_shift.size() * 4);
      device_buffers[s.old_ffn].download(v.ffn_shift.data(),
                                         v.ffn_shift.size() * 4);
      device_buffers[s.matrix].download(v.matrix.data(), v.matrix.size() * 4);
      out.layers.push_back(std::move(v));
    }
    return out;
  }
  Vector replay(int token, State &state, bool persistent = false) {
    if (device_failed)
      throw std::runtime_error(
          "Recreate resident graph after a device dispatch failure");
    const size_t c = weights.channels(), n = weights.head_size();
    if (token < 0 || size_t(token) >= weights.vocabulary() ||
        (!persistent && state.layers.size() != states.size()))
      throw std::runtime_error("Invalid graph token/state");
    auto bind = [&](Id id, const Vector &v) {
      if (v.size() != buffers[id].size)
        throw std::runtime_error("Graph state dimension mismatch");
      std::copy(v.begin(), v.end(), buffers[id].value.begin());
    };
    if (persistent && (!resident_state_valid || device_buffers.empty()))
      throw std::runtime_error("load_state required before resident replay");
    if (!persistent)
      for (size_t i = 0; i < states.size(); ++i) {
        auto &s = state.layers[i];
        if (s.attention_shift.size() != c || s.ffn_shift.size() != c ||
            s.matrix.size() != c * n)
          throw std::runtime_error("Graph state dimensions mismatch");
      }
    const auto &emb = weights.at("emb.weight").data;
    std::copy_n(emb.data() + size_t(token) * c, c,
                buffers[embedding].value.data());
    if (!persistent)
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
      resident_state_valid = false;
      if (!persistent)
        for (const auto &s : states) {
          upload(s.old_attention);
          upload(s.old_ffn);
          upload(s.matrix);
        }
      size_t node_index = 0;
      const bool profile = std::getenv("RWKV_XDNA_PROFILE") != nullptr;
      std::map<std::string, std::pair<size_t, double>> timings;
      for (size_t i = 0; i < runs.size(); ++i) {
        try {
          const auto start = profile ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
          runs[i].execute();
          if (profile) {
            const size_t owner = std::lower_bound(node_run_ends.begin(),
                                                  node_run_ends.end(), i + 1) -
                                 node_run_ends.begin();
            const auto &n = nodes[owner];
            const auto ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start)
                                .count();
            std::string category;
            if (n.kind == Kind::Recurrent)
              category = "wkv";
            else if (n.kind == Kind::Linear)
              category = n.transpose                         ? "rank"
                         : n.output == logits                ? "head"
                         : buffers[n.output].size == 8192    ? "ffn_key"
                         : buffers[n.inputs[0]].size == 8192 ? "ffn_value"
                                                             : "projection";
            else
              category = n.op == Op::Norm ? (n.group == 64 ? "finish" : "norm")
                         : n.op == Op::NormalizeKey ? "prepare"
                         : n.op == Op::Mix          ? "mix"
                                                    : "element";
            auto &t = timings[category];
            ++t.first;
            t.second += ms;
          }
          while (trace && node_index < nodes.size() &&
                 i + 1 == node_run_ends[node_index]) {
            const auto &node = nodes[node_index];
            auto &out = buffers[node.output].value;
            device_buffers[node.output].download(out.data(), out.size() * 4);
            Vector *matrix = nullptr;
            if (node.kind == Kind::Recurrent) {
              matrix = &buffers[node.inputs[0]].value;
              device_buffers[node.inputs[0]].download(matrix->data(),
                                                      matrix->size() * 4);
            }
            trace(node_index++, out, matrix);
          }
        } catch (const std::exception &e) {
          device_failed = true;
          const size_t failed_node =
              std::lower_bound(node_run_ends.begin(), node_run_ends.end(),
                               i + 1) -
              node_run_ends.begin();
          throw std::runtime_error(
              "Resident run " + std::to_string(i) + " node " +
              std::to_string(failed_node) + " kind " +
              std::to_string(static_cast<int>(nodes[failed_node].kind)) +
              " op " + std::to_string(static_cast<int>(nodes[failed_node].op)) +
              ": " + e.what());
        }
      }
      if (profile)
        for (const auto &entry : timings)
          std::cerr << "decode_profile " << entry.first
                    << " runs=" << entry.second.first
                    << " ms=" << entry.second.second << '\n';
      auto download = [&](Id id) {
        auto &v = buffers[id].value;
        device_buffers[id].download(v.data(), v.size() * 4);
      };
      download(logits);
      if (!persistent)
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
        if (trace)
          trace(&node - nodes.data(), read(node.output),
                node.kind == Kind::Recurrent ? &buffers[a[0]].value : nullptr);
      }
    for (float f : read(logits))
      if (!std::isfinite(f)) {
        if (!device_buffers.empty())
          device_failed = true;
        throw std::runtime_error("Nonfinite graph logits");
      }
    if (persistent) {
      try {
        for (auto &run : shift_runs)
          run.execute();
      } catch (...) {
        device_failed = true;
        throw;
      }
      resident_state_valid = true;
      ++replays;
      return read(logits);
    }
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
void DecodeGraph::load_state(const State &state) { impl_->load_state(state); }
State DecodeGraph::export_state() const { return impl_->export_state(); }
Vector DecodeGraph::replay_resident(int token) {
  State unused;
  return impl_->replay(token, unused, true);
}
void DecodeGraph::set_trace(Trace trace) { impl_->trace = std::move(trace); }
GraphStats DecodeGraph::stats() const {
  return {impl_->nodes.size(),
          impl_->buffers.size(),
          impl_->replays,
          impl_->runs.size(),
          impl_->resident_bytes,
          impl_->upload_bytes,
          impl_->download_bytes,
          impl_->root_bos,
          impl_->runs.size() + impl_->shift_runs.size(),
          impl_->weights.channels() * 4,
          impl_->weights.vocabulary() * 4};
}
} // namespace rwkv::inference

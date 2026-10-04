// SPDX-License-Identifier: Apache-2.0
#include "graph_internal.hpp"
#include "weight_layout.hpp"
#include <algorithm>
#include <stdexcept>

namespace rwkv::inference {
xdna::Session &DecodeGraph::Impl::session(const std::filesystem::path &root,
                                          const std::string &name) {
  auto &s = sessions[name];
  if (!s) {
    try {
      s = std::make_unique<xdna::Session>(root / name / "design.xclbin",
                                          root / name / "instructions.bin");
    } catch (const std::exception &e) {
      throw std::runtime_error("Create resident session " + name + " (" +
                               std::to_string(sessions.size()) +
                               " programs): " + e.what());
    }
  }
  return *s;
}
xdna::DeviceBuffer
DecodeGraph::Impl::initialized(xdna::Session &s, const Vector &v, bool shared) {
  auto b = s.allocate(v.size() * sizeof(float));
  b.upload(v.data(), v.size() * sizeof(float));
  resident_bytes += b.size();
  ++root_bos;
  if (capture_prefill && !shared)
    mutable_allocations.push_back({b, v});
  return b;
}
xdna::DeviceBuffer DecodeGraph::Impl::initialized_pair(xdna::Session &s,
                                                       const Vector &v) {
  if (!capture_prefill)
    return initialized(s, v);
  Vector pair = v;
  for (size_t t = 1; t < prefill_chunk_tokens; ++t)
    pair.insert(pair.end(), v.begin(), v.end());
  auto root = initialized(s, pair);
  mutable_allocations.back().token_bytes = v.size() * sizeof(float);
  return root.slice(0, v.size() * sizeof(float));
}
void DecodeGraph::Impl::append_run(xdna::Session &s,
                                   std::vector<xdna::DeviceBuffer> arguments,
                                   Stage stage) {
  runs.push_back(s.prepare(arguments));
  bindings.push_back({&s, std::move(arguments), stage});
}
xdna::DeviceBuffer DecodeGraph::Impl::initialized_bf16(xdna::Session &s,
                                                       const Vector &v) {
  auto half = weight_layout::to_bf16(v);
  auto b = s.allocate(half.size() * 2);
  b.upload(half.data(), half.size() * 2);
  resident_bytes += b.size();
  ++root_bos;
  return b;
}
// Allocate stable shared arenas before binding any run. The recurrence arena
// contains 27 FP32 vectors for layer zero, or 30 with fused value inputs;
// its aliases are also consumed by attention projection.
void DecodeGraph::Impl::prepare_resident_arenas(
    const std::filesystem::path &root, ResidentLayout &layout) {
  const size_t c = weights.channels();
  auto &value_args = layout.value_args;
  auto &recurrence_stages = layout.recurrence_stages;
  auto &allocator = session(root, "upstream-norm");
  device_buffers.resize(buffers.size());
  for (size_t ni = 0; ni < nodes.size(); ++ni) {
    const auto &node = nodes[ni];
    if (node.kind != Kind::Recurrent)
      continue;
    {
      if (ni < 6 || ni + 4 >= nodes.size())
        throw std::runtime_error("Incomplete recurrence stage");
      const auto *pre = &nodes[ni - 6];
      const auto *post = &nodes[ni + 1];
      const Op prep_ops[] = {Op::NormalizeKey, Op::Sigmoid,  Op::Negate,
                             Op::Multiply,     Op::KeyScale, Op::Decay};
      const Op post_ops[] = {Op::Norm, Op::Rkv, Op::Add, Op::Multiply};
      for (size_t j = 0; j < 6; ++j)
        if (pre[j].kind != Kind::Element || pre[j].op != prep_ops[j])
          throw std::runtime_error("Unexpected recurrence preparation");
      for (size_t j = 0; j < 4; ++j)
        if (post[j].kind != Kind::Element || post[j].op != post_ops[j])
          throw std::runtime_error("Unexpected recurrence finishing");
      if (pre[0].group != 64 || post[0].group != 64 ||
          post[0].epsilon != 64e-5f || post[0].inputs[0] != node.output ||
          pre[2].inputs[0] != pre[0].output ||
          pre[3].inputs[0] != pre[0].output ||
          pre[3].inputs[1] != pre[1].output ||
          pre[4].inputs[0] != pre[0].inputs[0] ||
          pre[4].inputs[1] != pre[1].output ||
          post[2].inputs[0] != post[0].output ||
          post[2].inputs[1] != post[1].output ||
          post[3].inputs[0] != post[2].output)
        throw std::runtime_error("Unexpected recurrence stage dependencies");
      const bool fused_value = ni >= 7 && nodes[ni - 7].kind == Kind::Element &&
                               nodes[ni - 7].op == Op::ValueResidual;
      std::vector<Id> ids(fused_value ? 30 : 27, none);
      if (fused_value) {
        const auto &value = nodes[ni - 7];
        if (value.output != node.inputs[4])
          throw std::runtime_error("Unexpected fused value dependency");
        ids[27] = value.inputs[0];
        ids[28] = value.inputs[2];
        ids[29] = value.inputs[3];
      }
      const std::vector<Id> pi{pre[0].inputs[0], pre[1].inputs[0],
                               pre[5].inputs[0], pre[0].inputs[1],
                               pre[4].inputs[2], pre[1].inputs[1],
                               pre[5].inputs[1]};
      std::copy(pi.begin(), pi.end(), ids.begin());
      const std::vector<Id> fi{node.output, post[0].inputs[1],
                               post[0].inputs[2], post[1].inputs[2],
                               post[3].inputs[1]};
      std::copy(fi.begin(), fi.end(), ids.begin() + 8);
      for (size_t j = 0; j < 6; ++j)
        ids[13 + j] = node.inputs[1 + j];
      ids[19] = pre[0].output;
      ids[20] = pre[1].output;
      for (size_t j = 0; j < 4; ++j)
        ids[21 + j] = post[j].output;
      Vector values(ids.size() * c, 0);
      for (size_t j = 0; j < ids.size(); ++j)
        if (ids[j] != none && buffers[ids[j]].constant)
          std::copy(read(ids[j]).begin(), read(ids[j]).end(),
                    values.begin() + j * c);
      auto arena = initialized_pair(allocator, values);
      for (size_t j = 0; j < ids.size(); ++j) {
        if (ids[j] == none)
          continue;
        if (device_buffers[ids[j]].size() || buffers[ids[j]].size != c)
          throw std::runtime_error("Unsupported recurrence arena alias");
        device_buffers[ids[j]] = arena.slice(j * c * 4, c * 4);
      }
      recurrence_stages.emplace(ni - 6, arena);
      continue;
    }
  }
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto *n = &nodes[i];
    if (n->kind != Kind::Element)
      continue;
    if (n->op == Op::ValueResidual) {
      auto input = recurrence_stages.at(i + 1).slice(27 * c * 4, (3 * c) * 4);
      value_args[i] = {input, device_buffers[n->inputs[1]]};
    }
  }
  // One root BO for constants and activation slots. Offsets stay fixed;
  // recurrent gather slots have their own six-vector backing allocation.
  size_t arena_floats = 0;
  for (size_t id = 0; id < buffers.size(); ++id)
    if (!device_buffers[id].size())
      arena_floats += ((buffers[id].size + (c - 1)) / c) * c;
  Vector arena_data(arena_floats, 0);
  size_t offset = 0;
  for (size_t id = 0; id < buffers.size(); ++id) {
    if (device_buffers[id].size())
      continue;
    const auto &b = buffers[id];
    if (b.constant)
      std::copy(b.constant->begin(), b.constant->end(),
                arena_data.begin() + offset);
    offset += ((b.size + (c - 1)) / c) * c;
  }
  auto arena = initialized(allocator, arena_data);
  offset = 0;
  for (size_t id = 0; id < buffers.size(); ++id) {
    if (device_buffers[id].size())
      continue;
    size_t size = ((buffers[id].size + (c - 1)) / c) * c;
    device_buffers[id] = arena.slice(offset * 4, size * 4);
    offset += size;
  }
}
} // namespace rwkv::inference

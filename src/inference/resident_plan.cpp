// SPDX-License-Identifier: Apache-2.0
#include "artifacts.hpp"
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
      throw std::runtime_error("Create resident session " + name + ": " +
                               e.what());
    }
  }
  return *s;
}
xdna::DeviceBuffer DecodeGraph::Impl::initialized(xdna::Session &s,
                                                  const Vector &v) {
  auto b = s.allocate(v.size() * sizeof(float));
  b.upload(v.data(), v.size() * sizeof(float));
  resident_bytes += b.size();
  ++root_bos;
  return b;
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
void DecodeGraph::Impl::prepare_resident(const std::filesystem::path &root) {
  validate_resident_artifacts(root, weights);
  ResidentLayout layout;
  prepare_resident_arenas(root, layout);
  prepare_resident_runs(root, layout);
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
// Allocate stable shared arenas before binding any run. The recurrence arena
// contains 27 FP32 vectors; its aliases are also consumed by attention
// projection.
void DecodeGraph::Impl::prepare_resident_arenas(
    const std::filesystem::path &root, ResidentLayout &layout) {
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
      std::vector<Id> ids(27, none);
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
      Vector values(27 * 2048, 0);
      for (size_t j = 0; j < ids.size(); ++j)
        if (ids[j] != none && buffers[ids[j]].constant)
          std::copy(read(ids[j]).begin(), read(ids[j]).end(),
                    values.begin() + j * 2048);
      auto arena = initialized(allocator, values);
      for (size_t j = 0; j < ids.size(); ++j) {
        if (ids[j] == none)
          continue;
        if (device_buffers[ids[j]].size() || buffers[ids[j]].size != 2048)
          throw std::runtime_error("Unsupported recurrence arena alias");
        device_buffers[ids[j]] = arena.slice(j * 2048 * 4, 2048 * 4);
      }
      recurrence_stages.emplace(ni - 6, arena);
      continue;
    }
  }
  auto pack_vectors = [&](const std::vector<Id> &ids, size_t slots) {
    Vector value(slots * 2048, 0);
    for (size_t i = 0; i < ids.size(); ++i) {
      if (buffers[ids[i]].constant)
        std::copy(read(ids[i]).begin(), read(ids[i]).end(),
                  value.begin() + i * 2048);
      if (device_buffers[ids[i]].size())
        throw std::runtime_error("Stage input alias");
    }
    auto packed = initialized(allocator, value);
    for (size_t i = 0; i < ids.size(); ++i)
      device_buffers[ids[i]] = packed.slice(i * 2048 * 4, 2048 * 4);
    return packed;
  };
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto *n = &nodes[i];
    if (n->kind != Kind::Element)
      continue;
    if (n->op == Op::ValueResidual) {
      auto input = pack_vectors({n->inputs[0], n->inputs[2], n->inputs[3]}, 3);
      value_args[i] = {input, device_buffers[n->inputs[1]],
                       device_buffers[n->output]};
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
  auto arena = initialized(allocator, arena_data);
  offset = 0;
  for (size_t id = 0; id < buffers.size(); ++id) {
    if (device_buffers[id].size())
      continue;
    size_t size = ((buffers[id].size + 2047) / 2048) * 2048;
    device_buffers[id] = arena.slice(offset * 4, size * 4);
    offset += size;
  }
}
// Match the recorded mathematical nodes to the one supported device schedule.
// Every logical node remains available to diagnostic traces after its stage
// runs.
void DecodeGraph::Impl::prepare_resident_runs(const std::filesystem::path &root,
                                              ResidentLayout &layout) {
  auto &value_args = layout.value_args;
  auto &recurrence_stages = layout.recurrence_stages;
  std::map<Id, xdna::DeviceBuffer> mixed_inputs;
  struct Branch {
    size_t first, last;
    int activation;
  };
  auto rank_branches = [&](size_t cursor) {
    std::vector<Branch> branches;
    for (int activation : {1, 0, 2, 0}) {
      if (cursor >= nodes.size() || nodes[cursor].kind != Kind::Linear ||
          !nodes[cursor].transpose ||
          buffers[nodes[cursor].inputs[0]].size != 2048 ||
          buffers[nodes[cursor].output].size > 256)
        break;
      size_t last = cursor + 1;
      if (activation) {
        if (last >= nodes.size() || nodes[last].kind != Kind::Element ||
            nodes[last].op != (activation == 1 ? Op::Tanh : Op::Sigmoid) ||
            nodes[last].inputs[0] != nodes[cursor].output)
          throw std::runtime_error("Unexpected batched rank activation");
        ++last;
      }
      if (last >= nodes.size() || nodes[last].kind != Kind::Linear ||
          !nodes[last].transpose ||
          nodes[last].inputs[0] != nodes[last - 1].output ||
          buffers[nodes[last].output].size != 2048)
        throw std::runtime_error("Unexpected batched rank output");
      branches.push_back({cursor, last, activation});
      cursor = last + 1;
    }
    return branches;
  };
  auto pack_rank = [&](const std::vector<Branch> &branches, size_t first_cores,
                       size_t second_cores,
                       const xdna::DeviceBuffer &auxiliary) {
    const size_t count = branches.size();
    std::vector<weight_layout::RankBranch> matrices;
    for (size_t p = 0; p < count; ++p) {
      const auto &branch = branches[p];
      const auto &first = nodes[branch.first];
      const auto &last = nodes[branch.last];
      device_buffers[first.output] = auxiliary.slice(p * 256 * 4, 256 * 4);
      if (branch.activation)
        device_buffers[nodes[branch.last - 1].output] =
            auxiliary.slice((count + p) * 256 * 4, 256 * 4);
      matrices.push_back({first.weight, last.weight});
    }
    auto packed =
        weight_layout::rank_batch(matrices, first_cores, second_cores);
    return packed;
  };
  for (size_t node_index = 0; node_index < nodes.size(); ++node_index) {
    const auto &node = nodes[node_index];
    const auto &a = node.inputs;
    if (node.kind == Kind::Element && node.op == Op::Norm &&
        node.group == 2048 && node.epsilon == 1e-5f &&
        node_index + 1 < nodes.size()) {
      size_t count = 0;
      const Id old = nodes[node_index + 1].inputs[1];
      while (node_index + 1 + count < nodes.size()) {
        const auto &mix = nodes[node_index + 1 + count];
        if (mix.kind != Kind::Element || mix.op != Op::Mix ||
            mix.inputs[0] != node.output || mix.inputs[1] != old)
          break;
        ++count;
      }
      if (count == 1 && node_index + 5 < nodes.size()) {
        const auto *n = &nodes[node_index];
        if (n[2].kind != Kind::Linear || n[2].transpose ||
            n[2].inputs[0] != n[1].output ||
            n[2].weight->shape != std::vector<size_t>{8192, 2048} ||
            n[3].kind != Kind::Element || n[3].op != Op::ReluSquared ||
            n[3].inputs[0] != n[2].output || n[4].kind != Kind::Linear ||
            n[4].transpose || n[4].inputs[0] != n[3].output ||
            n[4].weight->shape != std::vector<size_t>{2048, 8192} ||
            n[5].kind != Kind::Element || n[5].op != Op::Add ||
            n[5].inputs[0] != a[0] || n[5].inputs[1] != n[4].output ||
            !std::any_of(
                states.begin(), states.end(),
                [&](const StateBinding &s) { return s.old_ffn == old; }))
          throw std::runtime_error("Unexpected ChannelMix topology");
        auto &stage = session(root, "bf16-channel-mix");
        Vector parameters(6144);
        for (size_t j = 0; j < 3; ++j) {
          const auto &value = read(j < 2 ? a[j + 1] : n[1].inputs[2]);
          if (value.size() != 2048)
            throw std::runtime_error("Unexpected ChannelMix parameter shape");
          std::copy(value.begin(), value.end(), parameters.begin() + j * 2048);
        }
        auto packed = weight_layout::channel_mix(*n[2].weight, *n[4].weight);
        auto diagnostic = initialized(stage, Vector(22528, 0));
        auto result = initialized(stage, Vector(4096, 0));
        device_buffers[old] = diagnostic.slice(0, 2048 * 4);
        device_buffers[n[0].output] = device_buffers[old];
        device_buffers[n[1].output] = diagnostic.slice(4096 * 4, 2048 * 4);
        device_buffers[n[2].output] = diagnostic.slice(6144 * 4, 8192 * 4);
        device_buffers[n[3].output] = diagnostic.slice(14336 * 4, 8192 * 4);
        device_buffers[n[4].output] = result.slice(0, 2048 * 4);
        device_buffers[n[5].output] = result.slice(2048 * 4, 2048 * 4);
        runs.push_back(stage.prepare(
            {device_buffers[a[0]], initialized(stage, parameters),
             initialized_bf16(stage, packed), diagnostic, result}));
        for (size_t j = 0; j < 6; ++j)
          node_run_ends.push_back(runs.size());
        node_index += 5;
        continue;
      }
      const auto name = "fused-norm-mix-" + std::to_string(count);
      if (count == 6) {
        if (!std::any_of(
                states.begin(), states.end(),
                [&](const StateBinding &s) { return s.old_attention == old; }))
          throw std::runtime_error("Unexpected norm/mix shift binding");
        auto &stage = session(root, name);
        Vector parameters((2 + count) * 2048);
        for (size_t j = 0; j < 2 + count; ++j) {
          const auto &value =
              read(j < 2 ? a[1 + j] : nodes[node_index + j - 1].inputs[2]);
          if (value.size() != 2048)
            throw std::runtime_error("Unexpected norm/mix parameter shape");
          std::copy(value.begin(), value.end(), parameters.begin() + j * 2048);
        }
        auto pair = initialized(stage, Vector(4096, 0));
        auto mixed = initialized(stage, Vector(count * 2048, 0));
        device_buffers[node.output] = pair.slice(0, 2048 * 4);
        for (size_t j = 0; j < count; ++j)
          device_buffers[nodes[node_index + 1 + j].output] =
              mixed.slice(j * 2048 * 4, 2048 * 4);
        mixed_inputs.emplace(nodes[node_index + 1].output, mixed);
        mixed_inputs.emplace(nodes[node_index + 2].output, mixed);
        runs.push_back(
            stage.prepare({device_buffers[a[0]], initialized(stage, parameters),
                           device_buffers[old], pair, mixed}));
        for (size_t j = 0; j <= count; ++j)
          node_run_ends.push_back(runs.size());
        node_index += count;
        continue;
      }
    }
    if (node.kind == Kind::Linear && !node.transpose &&
        node.weight->shape == std::vector<size_t>{2048, 2048} &&
        node_index + 1 < nodes.size() &&
        nodes[node_index + 1].kind == Kind::Element &&
        nodes[node_index + 1].op == Op::Add &&
        nodes[node_index + 1].inputs[1] == node.output) {
      auto &stage = session(root, "bf16-projection-residual");
      auto packed =
          weight_layout::projection(*node.weight, false, 0, 2048, 2048);
      auto result = initialized(stage, Vector(4096, 0));
      device_buffers[node.output] = result.slice(0, 2048 * 4);
      device_buffers[nodes[node_index + 1].output] =
          result.slice(2048 * 4, 2048 * 4);
      runs.push_back(stage.prepare(
          {device_buffers[a[0]], initialized_bf16(stage, packed),
           device_buffers[nodes[node_index + 1].inputs[0]], result}));
      node_run_ends.push_back(runs.size());
      node_run_ends.push_back(runs.size());
      ++node_index;
      continue;
    }
    if (node.kind == Kind::Linear && !node.transpose &&
        mixed_inputs.count(a[0]) && node_index + 3 < nodes.size()) {
      const auto branches = rank_branches(node_index + 3);
      const size_t count = branches.size();
      const size_t cursor =
          branches.empty() ? node_index : branches.back().last + 1;
      const size_t prepare_index = cursor + (count == 4 ? 1 : 0);
      const auto name = "bf16-attention-projections-" + std::to_string(count);
      if ((count == 3 || count == 4) &&
          recurrence_stages.count(prepare_index)) {
        if (nodes[prepare_index].inputs[0] != nodes[node_index + 1].output ||
            nodes[prepare_index + 5].inputs[0] !=
                nodes[branches[0].last].output ||
            nodes[prepare_index + 1].inputs[0] !=
                nodes[branches[1].last].output ||
            nodes[prepare_index + 10].inputs[1] !=
                nodes[branches[2].last].output ||
            (count == 4 &&
             (nodes[cursor].op != Op::ValueResidual ||
              nodes[cursor].inputs[0] != nodes[node_index + 2].output ||
              nodes[cursor].inputs[2] != nodes[branches[3].last].output)))
          throw std::runtime_error(
              "Unexpected attention projection arena layout");
        auto &stage = session(root, name);
        auto auxiliary = initialized(stage, Vector(count * 512, 0));
        auto ranks = pack_rank(branches, 2, 4, auxiliary);
        std::array<const Tensor *, 3> projections;
        for (size_t p = 0; p < 3; ++p) {
          const auto &n = nodes[node_index + p];
          if (n.kind != Kind::Linear || n.transpose)
            throw std::runtime_error("Unexpected attention RKV topology");
          projections[p] = n.weight;
        }
        auto packed = weight_layout::rkv(projections);
        packed.insert(packed.end(), ranks.begin(), ranks.end());
        auto value_aux = count == 4 ? value_args.at(cursor)[0]
                                    : initialized(stage, Vector(6144, 0));
        runs.push_back(stage.prepare(
            {mixed_inputs.at(a[0]), initialized_bf16(stage, packed),
             recurrence_stages.at(prepare_index), value_aux, auxiliary}));
        for (size_t j = node_index; j < cursor; ++j)
          node_run_ends.push_back(runs.size());
        node_index = cursor - 1;
        continue;
      }
    }
    if (recurrence_stages.count(node_index)) {
      auto &stage = session(root, "fused-recurrence-stage");
      runs.push_back(
          stage.prepare({device_buffers[nodes[node_index + 6].inputs[0]],
                         recurrence_stages.at(node_index)}));
      for (size_t j = 0; j < 11; ++j)
        node_run_ends.push_back(runs.size());
      node_index += 10;
      continue;
    }
    if (node.kind == Kind::Element && node.op == Op::ValueResidual) {
      auto &value = session(root, "fused-value");
      runs.push_back(value.prepare(value_args.at(node_index)));
    } else if (node.kind == Kind::Element && node.op == Op::Norm &&
               node.group == 2048 && node.epsilon == 1e-5f) {
      auto &norm = session(root, "upstream-norm");
      runs.push_back(
          norm.prepare({device_buffers[a[0]], device_buffers[a[1]],
                        device_buffers[a[2]], device_buffers[node.output]}));
    } else if (node.kind == Kind::Element) {
      throw std::runtime_error("Element not covered by the production plan");
    } else if (node.kind == Kind::Linear) {
      if (node.transpose || node.output != logits ||
          node.weight->shape != std::vector<size_t>{65536, 2048})
        throw std::runtime_error(
            "Projection not covered by the BF16 production plan");
      auto &head = session(root, "bf16-array-gemv-2048-65536");
      auto packed =
          weight_layout::projection(*node.weight, false, 0, 65536, 2048);
      runs.push_back(
          head.prepare({device_buffers[a[0]], initialized_bf16(head, packed),
                        device_buffers[node.output]}));
    } else {
      throw std::runtime_error("Recurrence not covered by the production plan");
    }
    node_run_ends.push_back(runs.size());
  }
}
} // namespace rwkv::inference

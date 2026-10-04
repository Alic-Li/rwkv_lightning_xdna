// SPDX-License-Identifier: Apache-2.0
#include "artifacts.hpp"
#include "graph_internal.hpp"
#include "quantization.hpp"
#include "weight_layout.hpp"
#include <algorithm>
#include <stdexcept>

namespace rwkv::inference {
// Keep the original plan for node/projection tracing. Normal execution reuses
// its runs and BOs, replacing value-recurrence/output and the following FFN
// input binding. Replay adds no host transfer.
void DecodeGraph::Impl::prepare_decode_fusion(
    const std::filesystem::path &root) {
  const size_t c = weights.channels();
  if (states.size() < 2 || c != 2048 || weights.at("blocks.0.ffn.key.weight").shape[0] != 8192)
    return;
  check_artifact(root, "bf16-decode-recurrence-projection",
                 {{"schema_version", 1},
                  {"batch", 1},
                  {"channels", c},
                  {"head_size", 64},
                  {"arena_vectors", 30},
                  {"lanes", 7},
                  {"cores", 31},
                  {"output_vectors", {25, 26}},
                  {"dtype", "bfloat16"},
                  {"state_dtype", "float32"}});
  auto &fused = session(root, "bf16-decode-recurrence-projection");
  decode_runs.reserve(runs.size());
  decode_run_ends.reserve(node_run_ends.size());
  std::vector<size_t> ends(runs.size() + 1);
  xdna::DeviceBuffer ffn_input;
  for (size_t i = 0; i < bindings.size(); ++i) {
    const auto &binding = bindings[i];
    const auto &args = binding.arguments;
    if (binding.stage == Stage::Recurrence && args.size() == 3) {
      if (i + 2 >= bindings.size() || bindings[i + 1].stage != Stage::Output ||
          bindings[i + 2].stage != Stage::FFN)
        throw std::runtime_error("Unsupported decode fusion schedule");
      const auto &projection = bindings[i + 1].arguments;
      const auto offset = projection[0].offset_within(args[1]);
      if (!offset || *offset != 24 * c * 4 ||
          args[1].size() != 30 * c * 4)
        throw std::runtime_error("Unsupported decode fusion arena");
      decode_runs.push_back(fused.prepare(
          {args[0], args[1], args[2], projection[1], projection[2]}));
      ffn_input = args[1].slice(26 * c * 4, c * 4);
      ends[i + 1] = ends[i + 2] = decode_runs.size();
      ++i;
    } else {
      if (ffn_input.size()) {
        if (binding.stage != Stage::FFN)
          throw std::runtime_error("Missing fused decode FFN");
        auto remapped = args;
        remapped[0] = ffn_input;
        decode_runs.push_back(binding.session->prepare(remapped));
        ffn_input = {};
      } else
        decode_runs.push_back(runs[i]);
      ends[i + 1] = decode_runs.size();
    }
  }
  for (size_t end : node_run_ends)
    decode_run_ends.push_back(ends.at(end));
}
void DecodeGraph::Impl::prepare_resident(const std::filesystem::path &root) {
  const size_t c = weights.channels();
  validate_resident_artifacts(root, weights, weight_mode);
  if (capture_prefill)
    validate_prefill_artifacts(root, weight_mode, prefill_chunk_tokens);
  ResidentLayout layout;
  prepare_resident_arenas(root, layout);
  prepare_resident_runs(root, layout);
  prepare_decode_fusion(root);
  if (capture_prefill) {
    if (prefill_chunk_tokens == 4)
      prepare_chunk4(root);
    else
      prepare_prefill(root);
  }
  bindings.clear();
  bindings.shrink_to_fit();
  upload_bytes = c * 4;
  download_bytes = weights.vocabulary() * 4;
  for (const auto &s : states) {
    size_t bytes = (buffers[s.old_attention].size + buffers[s.old_ffn].size +
                    buffers[s.matrix].size) *
                   4;
    upload_bytes += bytes;
    download_bytes += bytes;
  }
}
// Match the recorded mathematical nodes to the one supported device schedule.
// Every logical node remains available to diagnostic traces after its stage
// runs.
void DecodeGraph::Impl::prepare_resident_runs(const std::filesystem::path &root,
                                              ResidentLayout &layout) {
  const size_t c = weights.channels(), hidden = weights.at("blocks.0.ffn.key.weight").shape[0];
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
          buffers[nodes[cursor].inputs[0]].size != c ||
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
          buffers[nodes[last].output].size != c)
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
        node.group == c && node.epsilon == 1e-5f &&
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
            n[2].weight->shape != std::vector<size_t>{hidden, c} ||
            n[3].kind != Kind::Element || n[3].op != Op::ReluSquared ||
            n[3].inputs[0] != n[2].output || n[4].kind != Kind::Linear ||
            n[4].transpose || n[4].inputs[0] != n[3].output ||
            n[4].weight->shape != std::vector<size_t>{c, hidden} ||
            n[5].kind != Kind::Element || n[5].op != Op::Add ||
            n[5].inputs[0] != a[0] || n[5].inputs[1] != n[4].output ||
            !std::any_of(
                states.begin(), states.end(),
                [&](const StateBinding &s) { return s.old_ffn == old; }))
          throw std::runtime_error("Unexpected ChannelMix topology");
        const bool int8 = weight_mode != WeightMode::BFloat16;
        auto &stage =
            session(root, int8 ? "int8-channel-mix" : "bf16-channel-mix");
        Vector parameters(3 * c);
        for (size_t j = 0; j < 3; ++j) {
          const auto &value = read(j < 2 ? a[j + 1] : n[1].inputs[2]);
          if (value.size() != c)
            throw std::runtime_error("Unexpected ChannelMix parameter shape");
          std::copy(value.begin(), value.end(), parameters.begin() + j * c);
        }
        xdna::DeviceBuffer packed_weights;
        if (int8) {
          auto packed = quantization::channel_mix(*n[2].weight, *n[4].weight);
          packed_weights = stage.allocate(packed.size());
          packed_weights.upload(packed.data(), packed.size());
          resident_bytes += packed.size();
          ++root_bos;
        } else {
          packed_weights = initialized_bf16(
              stage, weight_layout::channel_mix(*n[2].weight, *n[4].weight));
        }
        auto diagnostic = initialized(
            stage,
            Vector(capture_prefill ? (prefill_chunk_tokens == 4 ? 51200 : 26624)
                                   : 3 * c + 2 * hidden,
                   0));
        auto result = initialized(stage, Vector(2 * c, 0));
        device_buffers[old] = diagnostic.slice(0, c * 4);
        device_buffers[n[0].output] = device_buffers[old];
        device_buffers[n[1].output] = diagnostic.slice(2 * c * 4, c * 4);
        device_buffers[n[2].output] = diagnostic.slice(3 * c * 4, hidden * 4);
        device_buffers[n[3].output] = diagnostic.slice((3 * c + hidden) * 4, hidden * 4);
        device_buffers[n[4].output] = result.slice(0, c * 4);
        device_buffers[n[5].output] = result.slice(c * 4, c * 4);
        append_run(stage,
                   {device_buffers[a[0]], initialized(stage, parameters, true),
                    packed_weights, diagnostic, result},
                   Stage::FFN);
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
        Vector parameters((2 + count) * c);
        for (size_t j = 0; j < 2 + count; ++j) {
          const auto &value =
              read(j < 2 ? a[1 + j] : nodes[node_index + j - 1].inputs[2]);
          if (value.size() != c)
            throw std::runtime_error("Unexpected norm/mix parameter shape");
          std::copy(value.begin(), value.end(), parameters.begin() + j * c);
        }
        auto pair = initialized(stage, Vector(2 * c, 0));
        auto mixed = initialized_pair(stage, Vector(count * c, 0));
        device_buffers[node.output] = pair.slice(0, c * 4);
        for (size_t j = 0; j < count; ++j)
          device_buffers[nodes[node_index + 1 + j].output] =
              mixed.slice(j * c * 4, c * 4);
        mixed_inputs.emplace(nodes[node_index + 1].output, mixed);
        mixed_inputs.emplace(nodes[node_index + 2].output, mixed);
        append_run(stage,
                   {device_buffers[a[0]], initialized(stage, parameters, true),
                    device_buffers[old], pair, mixed},
                   Stage::Mix);
        for (size_t j = 0; j <= count; ++j)
          node_run_ends.push_back(runs.size());
        node_index += count;
        continue;
      }
    }
    if (node.kind == Kind::Linear && !node.transpose &&
        node.weight->shape == std::vector<size_t>{c, c} &&
        node_index + 1 < nodes.size() &&
        nodes[node_index + 1].kind == Kind::Element &&
        nodes[node_index + 1].op == Op::Add &&
        nodes[node_index + 1].inputs[1] == node.output) {
      auto &stage = session(root, "bf16-projection-residual");
      auto packed_weights = initialized_bf16(
          stage, weight_layout::projection(*node.weight, false, 0, c, c));
      auto result_root = initialized(
          stage,
          Vector(capture_prefill ? prefill_chunk_tokens * 2 * c : 2 * c, 0));
      auto result =
          capture_prefill ? result_root.slice(0, 2 * c * 4) : result_root;
      device_buffers[node.output] = result.slice(0, c * 4);
      device_buffers[nodes[node_index + 1].output] =
          result.slice(c * 4, c * 4);
      append_run(stage,
                 {device_buffers[a[0]], packed_weights,
                  device_buffers[nodes[node_index + 1].inputs[0]], result},
                 Stage::Output);
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
        auto auxiliary = initialized_pair(stage, Vector(count * 512, 0));
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
                                    : initialized_pair(stage, Vector(3 * c, 0));
        append_run(stage,
                   {mixed_inputs.at(a[0]), initialized_bf16(stage, packed),
                    recurrence_stages.at(prepare_index), value_aux, auxiliary},
                   Stage::Attention);
        for (size_t j = node_index; j < cursor; ++j)
          node_run_ends.push_back(runs.size());
        node_index = cursor - 1;
        continue;
      }
    }
    if (node.kind == Kind::Element && node.op == Op::ValueResidual) {
      const size_t prepare = node_index + 1;
      if (!recurrence_stages.count(prepare) ||
          nodes.at(prepare + 6).inputs[4] != node.output)
        throw std::runtime_error("Value residual not adjacent to recurrence");
      auto &stage = session(root, "fused-value-recurrence-stage");
      append_run(stage,
                 {device_buffers[nodes[prepare + 6].inputs[0]],
                  recurrence_stages.at(prepare), value_args.at(node_index)[1]},
                 Stage::Recurrence);
      for (size_t j = 0; j < 12; ++j)
        node_run_ends.push_back(runs.size());
      node_index += 11;
      continue;
    }
    if (recurrence_stages.count(node_index)) {
      auto &stage = session(root, "fused-recurrence-stage");
      append_run(stage,
                 {device_buffers[nodes[node_index + 6].inputs[0]],
                  recurrence_stages.at(node_index)},
                 Stage::Recurrence);
      for (size_t j = 0; j < 11; ++j)
        node_run_ends.push_back(runs.size());
      node_index += 10;
      continue;
    }
    if (node.kind == Kind::Element && node.op == Op::Norm &&
        node.group == c && node.epsilon == 1e-5f) {
      auto &norm = session(root, "upstream-norm");
      append_run(norm,
                 {device_buffers[a[0]], device_buffers[a[1]],
                  device_buffers[a[2]], device_buffers[node.output]},
                 Stage::Norm);
    } else if (node.kind == Kind::Element) {
      throw std::runtime_error("Element not covered by the production plan");
    } else if (node.kind == Kind::Linear) {
      if (node.transpose || node.output != logits ||
          node.weight->shape != std::vector<size_t>{weights.vocabulary(), c})
        throw std::runtime_error(
            "Projection not covered by the BF16 production plan");
      auto &head = session(root, "bf16-array-gemv-" + std::to_string(c) + "-" + std::to_string(weights.vocabulary()));
      auto packed =
          weight_layout::projection(*node.weight, false, 0, weights.vocabulary(), c);
      append_run(head,
                 {device_buffers[a[0]], initialized_bf16(head, packed),
                  device_buffers[node.output]},
                 Stage::Head);
    } else {
      throw std::runtime_error("Recurrence not covered by the production plan");
    }
    node_run_ends.push_back(runs.size());
  }
}
} // namespace rwkv::inference

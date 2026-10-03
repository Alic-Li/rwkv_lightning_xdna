// SPDX-License-Identifier: Apache-2.0
#include "graph_internal.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace rwkv::inference {
void DecodeGraph::Impl::validate_state(const State &state) const {
  if (state.layers.size() != states.size())
    throw std::runtime_error("Invalid state layers");
  for (const auto &s : state.layers)
    if (s.attention_shift.size() != weights.channels() ||
        s.ffn_shift.size() != weights.channels() ||
        s.matrix.size() != weights.channels() * weights.head_size())
      throw std::runtime_error("Invalid state dimensions");
}
void DecodeGraph::Impl::load_state(const State &state) {
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
State DecodeGraph::Impl::export_state() const {
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
Vector DecodeGraph::Impl::replay(int token, State &state, bool persistent) {
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
    const bool profile = std::getenv("RWKV_XDNA_PROFILE") != nullptr;
    const auto transfer_begin = profile
                                    ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{};
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
    const auto transfer_end = profile ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
    std::map<std::string, std::pair<size_t, double>> timings;
    std::map<int, std::pair<size_t, double>> layer_timings;
    double submit_us = 0, wait_us = 0;
    for (size_t i = 0; i < runs.size(); ++i) {
      try {
        const auto start = profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
        xdna::RunTiming run_timing;
        runs[i].execute(30000, profile ? &run_timing : nullptr);
        if (profile) {
          const size_t owner = std::lower_bound(node_run_ends.begin(),
                                                node_run_ends.end(), i + 1) -
                               node_run_ends.begin();
          const auto &n = nodes[owner];
          submit_us += run_timing.submit_us;
          wait_us += run_timing.wait_us;
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
          const size_t span = std::upper_bound(node_run_ends.begin(),
                                               node_run_ends.end(), i + 1) -
                              node_run_ends.begin() - owner;
          if (span > 1 && n.kind == Kind::Element && n.op == Op::Norm)
            category = span == 6 ? "channel_mix" : "norm_mix";
          else if (span > 1 && n.kind == Kind::Element &&
                   n.op == Op::ValueResidual)
            category = "value_recurrence_stage";
          else if (span > 1 && n.kind == Kind::Element &&
                   n.op == Op::NormalizeKey)
            category = "recurrence_stage";
          else if (span > 2 && n.kind == Kind::Linear && !n.transpose &&
                   buffers[n.output].size == 2048)
            category = "attention_projections";
          else if (span == 2 && n.kind == Kind::Linear && !n.transpose)
            category = "projection_residual";
          auto &t = timings[category];
          ++t.first;
          t.second += ms;
          auto &layer = layer_timings[n.layer];
          ++layer.first;
          layer.second += ms;
        }
        while ((trace || projection_trace) && node_index < nodes.size() &&
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
          if (projection_trace && node.kind == Kind::Linear) {
            Vector input(buffers[node.inputs[0]].size);
            device_buffers[node.inputs[0]].download(input.data(),
                                                    input.size() * 4);
            projection_trace(node_index, input, *node.weight, node.transpose,
                             out);
          }
          if (trace)
            trace(node_index, out, matrix);
          ++node_index;
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
            std::to_string(static_cast<int>(nodes[failed_node].kind)) + " op " +
            std::to_string(static_cast<int>(nodes[failed_node].op)) + ": " +
            e.what());
      }
    }
    if (profile)
      for (const auto &entry : timings)
        std::cerr << "decode_profile " << entry.first
                  << " runs=" << entry.second.first
                  << " ms=" << entry.second.second << '\n';
    if (profile) {
      for (const auto &entry : layer_timings)
        std::cerr << "decode_layer layer=" << entry.first
                  << " runs=" << entry.second.first
                  << " ms=" << entry.second.second << '\n';
      std::cerr << "decode_host submit_us=" << submit_us
                << " wait_us=" << wait_us << " upload_us="
                << std::chrono::duration<double, std::micro>(transfer_end -
                                                             transfer_begin)
                       .count()
                << '\n';
    }
    auto download = [&](Id id) {
      auto &v = buffers[id].value;
      device_buffers[id].download(v.data(), v.size() * 4);
    };
    const auto download_begin = profile
                                    ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{};
    download(logits);
    if (profile)
      std::cerr << "decode_logits download_us="
                << std::chrono::duration<double, std::micro>(
                       std::chrono::steady_clock::now() - download_begin)
                       .count()
                << '\n';
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
        result = backend->element(node.op, read(a[0]), read(a[1]), read(a[2]),
                                  read(a[3]), node.group, node.epsilon);
      else if (node.kind == Kind::Linear)
        result = backend->linear(read(a[0]), *node.weight, node.transpose);
      else
        result = backend->step(buffers[a[0]].value, read(a[1]), read(a[2]),
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
} // namespace rwkv::inference

// SPDX-License-Identifier: Apache-2.0
#include "graph_internal.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace rwkv::inference {
namespace {
bool same_region(const xdna::DeviceBuffer &a, const xdna::DeviceBuffer &b) {
  auto offset = a.offset_within(b);
  return offset && *offset == 0 && a.size() == b.size();
}
} // namespace

void DecodeGraph::Impl::prepare_prefill(const std::filesystem::path &root) {
  // Capture is setup-only. New buffers below must not extend the source list.
  capture_prefill = false;
  const size_t layers = states.size();
  if (bindings.size() != 5 * layers + 3 || bindings.front().stage != Stage::Norm ||
      bindings[bindings.size() - 2].stage != Stage::Norm || bindings.back().stage != Stage::Head)
    throw std::runtime_error("Unsupported prefill stage schedule");
  const Stage expected[] = {Stage::Mix, Stage::Attention, Stage::Recurrence, Stage::Output, Stage::FFN};
  for (size_t l = 0; l < layers; ++l)
    for (size_t i = 0; i < 5; ++i)
      if (bindings[1 + l * 5 + i].stage != expected[i])
        throw std::runtime_error("Unsupported prefill layer schedule");
  auto allocation = [&](const xdna::DeviceBuffer &view) -> xdna::DeviceBuffer {
    for (const auto &a : mutable_allocations)
      if (view.offset_within(a.root)) return a.root;
    throw std::runtime_error("Missing mutable prefill allocation");
  };
  std::vector<xdna::DeviceBuffer> paired_outputs, old_ffn_outputs, ffn_arenas;
  for (size_t l = 0; l < layers; ++l) {
    auto &projection = bindings[1 + l * 5 + 3].arguments;
    auto &ffn = bindings[1 + l * 5 + 4].arguments;
    auto pair = allocation(projection.at(3));
    if (pair.size() != 8192 * 4 || ffn.at(3).size() != 26624 * 4)
      throw std::runtime_error("Invalid prefill paired arena size");
    paired_outputs.push_back(pair);
    old_ffn_outputs.push_back(ffn.at(4).slice(2048 * 4, 2048 * 4));
    ffn_arenas.push_back(ffn.at(3));
  }
  struct Clone { xdna::DeviceBuffer original, copy; };
  std::vector<Clone> clones;
  auto &allocator = *bindings.front().session;
  auto remap = [&](const xdna::DeviceBuffer &view, bool second) -> xdna::DeviceBuffer {
    for (size_t l = 0; l < layers; ++l)
      if (same_region(view, old_ffn_outputs[l]))
        return ffn_arenas[l].slice((22528 + (second ? 2048 : 0)) * 4, 2048 * 4);
    if (!second) return view;
    // These are request state, shared and updated in temporal order by both
    // tokens. Other aliases of the activation arena get separate storage.
    for (const auto &s : states)
      if (same_region(view, device_buffers[s.old_attention]) ||
          same_region(view, device_buffers[s.matrix])) return view;
    for (const auto &pair : paired_outputs)
      if (auto offset = view.offset_within(pair)) {
        if (*offset + view.size() > 4096 * 4)
          throw std::runtime_error("Projection view spans both tokens");
        return pair.slice(*offset + 4096 * 4, view.size());
      }
    for (const auto &a : mutable_allocations)
      if (auto offset = view.offset_within(a.root)) {
        for (const auto &clone : clones)
          if (same_region(clone.original, a.root))
            return clone.copy.slice(*offset, view.size());
        auto copy = initialized(allocator, a.initial);
        clones.push_back({a.root, copy});
        return copy.slice(*offset, view.size());
      }
    // Matrix weights and packed parameters were never classified as mutable.
    return view;
  };
  auto bind = [&](size_t index, bool second) {
    const auto &binding = bindings.at(index);
    auto args = binding.arguments;
    for (auto &arg : args) arg = remap(arg, second);
    return binding.session->prepare(args);
  };
  prefill_embedding = remap(device_buffers[embedding], true);
  prefill_logits = remap(device_buffers[logits], true);
  prefill_body.push_back(bind(0, false));
  prefill_body.push_back(bind(0, true));
  auto &ffn = session(root, "bf16-prefill-ffn-b2-projection-input");
  for (size_t l = 0; l < layers; ++l) {
    for (size_t stage = 0; stage < 4; ++stage) {
      prefill_body.push_back(bind(1 + l * 5 + stage, false));
      prefill_body.push_back(bind(1 + l * 5 + stage, true));
    }
    const auto &args = bindings[1 + l * 5 + 4].arguments;
    auto half = initialized(ffn, Vector(10240, 0));
    prefill_body.push_back(ffn.prepare({paired_outputs[l], args[1], args[2], ffn_arenas[l], half}));
  }
  prefill_head.push_back(bind(bindings.size() - 2, true));
  prefill_head.push_back(bind(bindings.size() - 1, true));
  // Run objects retain all device storage. Release setup-only host snapshots.
  mutable_allocations.clear();
  mutable_allocations.shrink_to_fit();
  bindings.clear();
  bindings.shrink_to_fit();
}

Vector DecodeGraph::Impl::prefill(const std::vector<int> &tokens) {
  if (prefill_body.empty()) throw std::runtime_error("Construct graph with Batched2 prefill enabled");
  if (tokens.empty()) return {};
  if (!resident_state_valid || device_failed) throw std::runtime_error("No valid resident state");
  if (trace || projection_trace || std::getenv("RWKV_XDNA_PROFILE"))
    throw std::runtime_error("Disable trace/profile hooks for batched prefill");
  for (int token : tokens)
    if (token < 0 || size_t(token) >= weights.vocabulary())
      throw std::runtime_error("Invalid prefill token");
  const size_t c = weights.channels();
  const auto &emb = weights.at("emb.weight").data;
  Vector result;
  try {
    for (size_t t = 0; t + 1 < tokens.size(); t += 2) {
      resident_state_valid = false;
      device_buffers[embedding].upload(emb.data() + size_t(tokens[t]) * c, c * 4);
      prefill_embedding.upload(emb.data() + size_t(tokens[t + 1]) * c, c * 4);
      for (auto &run : prefill_body) { run.execute(); ++prefill_run_count; }
      if (t + 2 == tokens.size()) {
        for (auto &run : prefill_head) { run.execute(); ++prefill_run_count; }
        result.resize(weights.vocabulary());
        prefill_logits.download(result.data(), result.size() * 4);
        for (float x : result)
          if (!std::isfinite(x)) throw std::runtime_error("Nonfinite prefill logits");
      }
      replays += 2;
      resident_state_valid = true;
    }
    if (tokens.size() % 2) {
      State unused;
      result = replay(tokens.back(), unused, true);
      prefill_run_count += runs.size();
    }
  } catch (...) {
    resident_state_valid = false;
    device_failed = true;
    throw;
  }
  return result;
}
} // namespace rwkv::inference

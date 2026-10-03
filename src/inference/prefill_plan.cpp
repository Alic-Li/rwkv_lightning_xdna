// SPDX-License-Identifier: Apache-2.0
#include "graph_internal.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>
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
  auto paired_view = [&](const xdna::DeviceBuffer &view) -> xdna::DeviceBuffer {
    for (const auto &a : mutable_allocations)
      if (auto offset = view.offset_within(a.root)) {
        if (!a.token_bytes || *offset + view.size() > a.token_bytes)
          throw std::runtime_error("Invalid paired attention allocation");
        return a.root.slice(*offset, a.token_bytes + view.size());
      }
    throw std::runtime_error("Missing paired attention allocation");
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
        if (a.token_bytes) {
          if (*offset + view.size() > a.token_bytes)
            throw std::runtime_error("Activation view spans both tokens");
          return a.root.slice(*offset + a.token_bytes, view.size());
        }
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
    return PrefillRun{binding.session->prepare(args), binding.stage, second ? 1 : 0};
  };
  prefill_embedding = remap(device_buffers[embedding], true);
  prefill_logits = remap(device_buffers[logits], true);
  prefill_body.push_back(bind(0, false));
  prefill_body.push_back(bind(0, true));
  auto &ffn = session(root, "bf16-prefill-ffn-b2-projection-input");
  for (size_t l = 0; l < layers; ++l) {
    const bool fuse_projection = l > 0;
    auto ffn_input = paired_outputs[l];
    for (size_t stage = 0; stage < 4; ++stage) {
      const size_t index = 1 + l * 5 + stage;
      if (stage == 1) {
        const auto &args = bindings[index].arguments;
        const size_t branches = args[4].size() / (512 * sizeof(float));
        if (branches != 3 && branches != 4)
          throw std::runtime_error("Unsupported paired attention branch count");
        auto &attention = session(root, "bf16-attention-projections-" + std::to_string(branches) + "-b2");
        prefill_body.push_back({attention.prepare({paired_view(args[0]), args[1],
            paired_view(args[2]), paired_view(args[3]), paired_view(args[4])}), Stage::Attention, 2});
      } else if (stage == 2 && fuse_projection) {
        const auto &args = bindings[index].arguments;
        const auto &projection = bindings[index + 1].arguments;
        auto residual = ffn_arenas[l - 1].slice(22528 * 4, 4096 * 4);
        if (!same_region(remap(projection[2], false), residual.slice(0, 8192)) ||
            !same_region(remap(projection[2], true), residual.slice(8192, 8192)))
          throw std::runtime_error("Unsupported fused projection residual layout");
        auto arena = paired_view(args[1]);
        auto &fused = session(root, "bf16-prefill-recurrence-projection-b2");
        prefill_body.push_back({fused.prepare({args[0], arena, paired_view(args[2]),
            projection[1], residual}), Stage::RecurrenceOutput, 2});
        ffn_input = arena.slice(25 * 2048 * 4, 65536 * 4);
      } else if (stage == 2) {
        const auto &args = bindings[index].arguments;
        const bool value = args.size() == 3;
        auto &recurrence = session(root, value ? "prefill-value-recurrence-b2" : "prefill-recurrence-b2");
        std::vector<xdna::DeviceBuffer> pair{args[0], paired_view(args[1])};
        if (value) pair.push_back(paired_view(args[2]));
        prefill_body.push_back({recurrence.prepare(pair), Stage::Recurrence, 2});
      } else if (stage == 3) {
        if (fuse_projection) continue;
        const auto &args = bindings[index].arguments;
        auto input = paired_view(args[0]);
        const size_t stride = input.size() / sizeof(float) - 2048;
        auto &output = session(root, std::string("bf16") +
            "-prefill-output-b2-s" + std::to_string(stride));
        prefill_body.push_back({output.prepare({input, args[1], remap(args[2], false),
            remap(args[2], true), paired_outputs[l]}), Stage::Output, 2});
      } else {
        prefill_body.push_back(bind(index, false));
        prefill_body.push_back(bind(index, true));
      }
    }
    const auto &args = bindings[1 + l * 5 + 4].arguments;
    auto &ffn_stage = fuse_projection ? session(root, "bf16-prefill-ffn-b2-recurrence-input") : ffn;
    auto half = initialized(ffn_stage, Vector(10240, 0));
    prefill_body.push_back({ffn_stage.prepare({ffn_input, args[1], args[2], ffn_arenas[l], half}),
                            Stage::FFN, 2});
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
  if (prefill_body.empty()) throw std::runtime_error("Construct graph with batched prefill enabled");
  if (tokens.empty()) return {};
  if (!resident_state_valid || device_failed) throw std::runtime_error("No valid resident state");
  if (trace || projection_trace)
    throw std::runtime_error("Disable trace hooks for batched prefill");
  for (int token : tokens)
    if (token < 0 || size_t(token) >= weights.vocabulary())
      throw std::runtime_error("Invalid prefill token");
  const size_t c = weights.channels();
  const auto &emb = weights.at("emb.weight").data;
  Vector result;
  const bool profile = std::getenv("RWKV_XDNA_PROFILE") != nullptr;
  using Clock = std::chrono::steady_clock;
  auto elapsed_us = [](Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
  };
  struct Timing { size_t runs = 0; double elapsed = 0, submit = 0, wait = 0; };
  std::map<std::pair<Stage, int>, Timing> timings;
  double upload_us = 0, download_us = 0;
  auto execute = [&](PrefillRun &entry) {
    if (!profile) entry.run.execute();
    else {
      xdna::RunTiming measured;
      const auto start = Clock::now();
      entry.run.execute(30000, &measured);
      const double elapsed = elapsed_us(start);
      auto &t = timings[{entry.stage, entry.token_slot}];
      ++t.runs;
      t.elapsed += elapsed;
      t.submit += measured.submit_us;
      t.wait += measured.wait_us;
    }
    ++prefill_run_count;
  };
  try {
    for (size_t t = 0; t + prefill_chunk_tokens <= tokens.size(); t += prefill_chunk_tokens) {
      resident_state_valid = false;
      const auto upload_start = profile ? Clock::now() : Clock::time_point{};
      device_buffers[embedding].upload(emb.data() + size_t(tokens[t]) * c, c * 4);
      if (prefill_chunk_tokens == 2)
        prefill_embedding.upload(emb.data() + size_t(tokens[t + 1]) * c, c * 4);
      else for (size_t slot = 1; slot < prefill_chunk_tokens; ++slot)
        chunk_embeddings[slot - 1].upload(emb.data() + size_t(tokens[t + slot]) * c, c * 4);
      if (profile) upload_us += elapsed_us(upload_start);
      for (auto &run : prefill_body) execute(run);
      if (t + prefill_chunk_tokens == tokens.size()) {
        for (auto &run : prefill_head) execute(run);
        result.resize(weights.vocabulary());
        const auto download_start = profile ? Clock::now() : Clock::time_point{};
        prefill_logits.download(result.data(), result.size() * 4);
        if (profile) download_us += elapsed_us(download_start);
        for (float x : result)
          if (!std::isfinite(x)) throw std::runtime_error("Nonfinite prefill logits");
      }
      replays += prefill_chunk_tokens;
      resident_state_valid = true;
    }
    for (size_t t = tokens.size() - tokens.size() % prefill_chunk_tokens; t < tokens.size(); ++t) {
      State unused;
      result = replay(tokens[t], unused, true);
      prefill_run_count += decode_run_count();
    }
  } catch (...) {
    resident_state_valid = false;
    device_failed = true;
    throw;
  }
  if (profile) {
    auto name = [](Stage stage) {
      switch (stage) {
      case Stage::Norm: return "norm";
      case Stage::Mix: return "norm_mix";
      case Stage::Attention: return "attention_projections";
      case Stage::Recurrence: return "recurrence";
      case Stage::RecurrenceOutput: return "recurrence_projection_residual";
      case Stage::Output: return "projection_residual";
      case Stage::FFN: return "channel_mix";
      case Stage::Head: return "head";
      }
      return "unknown";
    };
    nlohmann::json rows = nlohmann::json::array();
    for (const auto &[key, t] : timings)
      rows.push_back({{"stage", name(key.first)}, {"token_slot", key.second},
          {"runs", t.runs}, {"elapsed_us", t.elapsed},
          {"submit_us", t.submit}, {"wait_us", t.wait}});
    std::cerr << "prefill_profile " << nlohmann::json({
        {"prompt_tokens", tokens.size()}, {"pairs", prefill_chunk_tokens == 2 ? tokens.size() / 2 : 0},
        {"chunk_tokens", prefill_chunk_tokens}, {"chunks", tokens.size() / prefill_chunk_tokens},
        {"odd_tail_runs", prefill_chunk_tokens == 2 ? (tokens.size() % 2) * decode_run_count() : 0},
        {"tail_runs", (tokens.size() % prefill_chunk_tokens) * decode_run_count()},
        {"chunk_upload_us", upload_us}, {"final_chunk_download_us", download_us},
        {"pair_upload_us", upload_us}, {"final_pair_download_us", download_us},
        {"stages", rows},
        {"scope", "Host stage wall time including scheduling, program switches, DMA and compute; tail tokens have separate decode_profile output."}
    }).dump() << '\n';
  }
  return result;
}
} // namespace rwkv::inference

// SPDX-License-Identifier: Apache-2.0
#include "graph_internal.hpp"
#include <stdexcept>

namespace rwkv::inference {
namespace {
bool same(const xdna::DeviceBuffer &a, const xdna::DeviceBuffer &b) {
  const auto offset = a.offset_within(b);
  return offset && *offset == 0 && a.size() == b.size();
}
}
// Four tokens advance layer by layer. Existing pair attention/FP32 recurrence
// kernels share state in time order, then one FFN reuses weights across all four.
void DecodeGraph::Impl::prepare_chunk4(const std::filesystem::path &root) {
  capture_prefill = false;
  const size_t layers = states.size();
  if (bindings.size() != 5 * layers + 3 || bindings.front().stage != Stage::Norm ||
      bindings[bindings.size()-2].stage != Stage::Norm || bindings.back().stage != Stage::Head)
    throw std::runtime_error("Unsupported chunk4 schedule");
  const Stage expected[] = {Stage::Mix, Stage::Attention, Stage::Recurrence, Stage::Output, Stage::FFN};
  for (size_t l = 0; l < layers; ++l)
    for (size_t j = 0; j < 5; ++j)
      if (bindings[1 + 5*l+j].stage != expected[j])
        throw std::runtime_error("Unsupported chunk4 layer schedule");
  auto allocation = [&](const xdna::DeviceBuffer &v) {
    for (const auto &a : mutable_allocations) if (v.offset_within(a.root)) return a.root;
    throw std::runtime_error("Missing chunk allocation");
  };
  std::vector<xdna::DeviceBuffer> projections, arenas, old_outputs;
  for (size_t l = 0; l < layers; ++l) {
    const auto &f = bindings[1+5*l+4].arguments;
    auto projection = allocation(bindings[1+5*l+3].arguments[3]);
    if (projection.size() != 16384*4 || f[3].size() != 51200*4)
      throw std::runtime_error("Invalid chunk4 arena size");
    projections.push_back(projection);
    arenas.push_back(f[3]);
    old_outputs.push_back(f[4].slice(2048*4, 2048*4));
  }
  struct Clone { xdna::DeviceBuffer original, copy; size_t slot; };
  std::vector<Clone> clones;
  auto &allocator = *bindings.front().session;
  auto remap = [&](const xdna::DeviceBuffer &v, size_t slot) -> xdna::DeviceBuffer {
    for (size_t l = 0; l < layers; ++l)
      if (same(v, old_outputs[l])) return arenas[l].slice((43008+slot*2048)*4, 2048*4);
    if (!slot) return v;
    for (const auto &s : states)
      if (same(v, device_buffers[s.old_attention]) || same(v, device_buffers[s.matrix])) return v;
    for (const auto &p : projections)
      if (auto offset = v.offset_within(p)) {
        if (*offset + v.size() > 4096*4) throw std::runtime_error("Invalid chunk projection view");
        return p.slice(*offset + slot*4096*4, v.size());
      }
    for (const auto &a : mutable_allocations)
      if (auto offset = v.offset_within(a.root)) {
        if (a.token_bytes) {
          if (*offset + v.size() > a.token_bytes) throw std::runtime_error("Invalid chunk activation view");
          return a.root.slice(*offset + slot*a.token_bytes, v.size());
        }
        for (const auto &clone : clones)
          if (clone.slot == slot && same(clone.original, a.root)) return clone.copy.slice(*offset, v.size());
        auto copy = initialized(allocator, a.initial);
        clones.push_back({a.root, copy, slot});
        return copy.slice(*offset, v.size());
      }
    return v; // Immutable weights and parameters.
  };
  auto pair_view = [&](const xdna::DeviceBuffer &v, size_t pair) {
    for (const auto &a : mutable_allocations)
      if (auto offset = v.offset_within(a.root)) {
        if (!a.token_bytes || *offset + v.size() > a.token_bytes)
          throw std::runtime_error("Invalid chunk pair view");
        return a.root.slice(*offset + pair*2*a.token_bytes, a.token_bytes+v.size());
      }
    throw std::runtime_error("Missing chunk pair view");
  };
  auto bind = [&](size_t index, size_t slot) {
    const auto &b = bindings.at(index);
    auto args = b.arguments;
    for (auto &a : args) a = remap(a, slot);
    return PrefillRun{b.session->prepare(args), b.stage, int(slot)};
  };
  for (size_t slot = 1; slot < 4; ++slot) chunk_embeddings.push_back(remap(device_buffers[embedding], slot));
  prefill_logits = remap(device_buffers[logits], 3);
  for (size_t slot = 0; slot < 4; ++slot) prefill_body.push_back(bind(0, slot));
  auto &ffn = session(root, std::string("bf16") +
                      "-prefill-ffn-b4-projection-input");
  for (size_t l = 0; l < layers; ++l) {
    std::array<std::vector<PrefillRun>, 4> stages;
    for (size_t pair = 0; pair < 2; ++pair) {
      const size_t base = 1+5*l;
      stages[0].push_back(bind(base, pair*2));
      stages[0].push_back(bind(base, pair*2+1));
      const auto &a = bindings[base+1].arguments;
      const auto branches = a[4].size()/(512*4);
      if (branches != 3 && branches != 4) throw std::runtime_error("Invalid chunk attention branches");
      auto &att = session(root, "bf16-attention-projections-"+std::to_string(branches)+"-b2");
      stages[1].push_back({att.prepare({pair_view(a[0],pair), a[1], pair_view(a[2],pair),
          pair_view(a[3],pair), pair_view(a[4],pair)}), Stage::Attention, int(pair*2)});
      const auto &r = bindings[base+2].arguments;
      const bool value = r.size() == 3;
      auto &rec = session(root, value ? "prefill-value-recurrence-b2" : "prefill-recurrence-b2");
      std::vector<xdna::DeviceBuffer> args{r[0], pair_view(r[1],pair)};
      if (value) args.push_back(pair_view(r[2],pair));
      stages[2].push_back({rec.prepare(args), Stage::Recurrence, int(pair*2)});
      const auto &o = bindings[base+3].arguments;
      auto input = pair_view(o[0],pair);
      const auto stride = input.size()/4-2048;
      auto &out = session(root, std::string("bf16") +
                          "-prefill-output-b2-s"+std::to_string(stride));
      stages[3].push_back({out.prepare({input,o[1],remap(o[2],pair*2),remap(o[2],pair*2+1),
          projections[l].slice(pair*8192*4,8192*4)}),Stage::Output,int(pair*2)});
    }
    // Keep each program active across the chunk. Mutable attention/state
    // advances in token order; activation slots remain disjoint.
    for (auto &stage : stages)
      for (auto &run : stage) prefill_body.push_back(std::move(run));
    const auto &f = bindings[1+5*l+4].arguments;
    auto half = initialized(ffn, Vector(20480,0));
    prefill_body.push_back({ffn.prepare({projections[l],f[1],f[2],arenas[l],half}),Stage::FFN,4});
  }
  prefill_head.push_back(bind(bindings.size()-2,3));
  prefill_head.push_back(bind(bindings.size()-1,3));
  mutable_allocations.clear();
  mutable_allocations.shrink_to_fit();
  bindings.clear();
  bindings.shrink_to_fit();
}
} // namespace rwkv::inference

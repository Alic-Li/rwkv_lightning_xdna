// SPDX-License-Identifier: Apache-2.0
#include "artifacts.hpp"
#include <fstream>
#include <stdexcept>
namespace rwkv::inference {
void check_artifact(const std::filesystem::path &root, const std::string &name,
                    const nlohmann::json &expected) {
  std::ifstream input(root / name / "config.json");
  if (!input)
    throw std::runtime_error("Missing production artifact: " + name);
  nlohmann::json config;
  input >> config;
  if (config.value("trace_buffer_bytes", size_t(0)) != 0)
    throw std::runtime_error(
        "Diagnostic trace artifact cannot be used for inference: " + name);
  if (config.contains("experimental") ||
      config.value("mode_attention", false) ||
      (config.value("weight_layout", "row_major") != "row_major" &&
       config.value("weight_layout", "row_major") != "w2_k_major_4"))
    throw std::runtime_error(
        "Experimental artifact cannot be used for inference: " + name);
  for (auto it = expected.begin(); it != expected.end(); ++it)
    if (!config.contains(it.key()) || config.at(it.key()) != it.value())
      throw std::runtime_error("Incompatible production artifact: " + name +
                               " field " + it.key());
  for (const auto *file : {"design.xclbin", "instructions.bin"})
    if (!std::filesystem::is_regular_file(root / name / file))
      throw std::runtime_error("Missing device binary: " +
                               (root / name / file).string());
}
void validate_resident_artifacts(const std::filesystem::path &root,
                                 const Weights &w, WeightMode mode) {
  if (w.channels() != 2048 || w.heads() != 32 || w.vocabulary() != 65536)
    throw std::runtime_error(
        "BF16 NPU inference requires C=2048, heads=32, vocabulary=65536");
  check_artifact(root, "upstream-norm",
                 {{"schema_version", 1},
                  {"dtype", "float32"},
                  {"channels", 2048},
                  {"epsilon", 1e-5}});
  check_artifact(root, "fused-norm-mix-6",
                 {{"schema_version", 1},
                  {"dtype", "float32"},
                  {"channels", 2048},
                  {"mixes", 6},
                  {"exact_fp32", false}});
  check_artifact(root, "fused-value-recurrence-stage",
                 {{"schema_version", 1},
                  {"dtype", "float32"},
                  {"channels", 2048},
                  {"head_size", 64},
                  {"arena_vectors", 30},
                  {"lanes", 7},
                  {"fused_value", true},
                  {"exact_fp32", false}});
  check_artifact(root, "fused-recurrence-stage",
                 {{"schema_version", 1},
                  {"dtype", "float32"},
                  {"channels", 2048},
                  {"head_size", 64},
                  {"arena_vectors", 27},
                  {"exact_fp32", false}});
  if (mode != WeightMode::BFloat16)
    check_artifact(
        root, "int8-channel-mix",
        {{"schema_version", 1},
         {"dtype", "int8"},
         {"channels", 2048},
         {"hidden", 8192},
         {"key_cores", 4},
         {"value_cores", 4},
         {"tile_bytes", 4160},
         {"quantization", "symmetric_per_output_127"},
         {"scale", "fp16_expanded_fp32"},
         {"activation_dtype", "int8"},
         {"accumulator_dtype", "int32"},
         {"weight_chunk_tiles", 1},
         {"weight_layout", "w2_k_major_4"},
         {"activation_placement", "w1_stream256"},
         {"w1_output_partition", "block_cyclic_256"},
         {"diagnostic_layout", "raw_active_block_cyclic_scatter"},
         {"integer_reduction", "row_reduce"},
         {"activation_block", 256},
         {"activation_packet_bytes", 320},
         {"multiply_dtype", "int8"},
         {"partial_sum_dtype", "float32"},
         {"activation_conversion", "dynamic_symmetric_block256_nearest_even"}});
  else
    check_artifact(root, "bf16-channel-mix",
                   {{"schema_version", 1},
                    {"dtype", "bfloat16"},
                    {"channels", 2048},
                    {"hidden", 8192},
                    {"key_cores", 4},
                    {"value_cores", 4},
                    {"weight_layout", "row_major"},
                    {"exact_fp32", false}});
  check_artifact(root, "bf16-projection-residual",
                 {{"schema_version", 1},
                  {"dtype", "bfloat16"},
                  {"channels", 2048},
                  {"cores", 11},
                  {"exact_fp32", false}});
  for (int branches : {3, 4})
    check_artifact(root,
                   "bf16-attention-projections-" + std::to_string(branches),
                   {{"schema_version", 1},
                    {"dtype", "bfloat16"},
                    {"channels", 2048},
                    {"rank", 256},
                    {"branches", branches},
                    {"cores", 14},
                    {"exact_fp32", false}});
  check_artifact(root, "bf16-array-gemv-2048-65536",
                 {{"schema_version", 1},
                  {"dtype", "bfloat16"},
                  {"rows", 65536},
                  {"cores", 8},
                  {"k", 2048}});
}
void validate_prefill_artifacts(const std::filesystem::path &root,
                                WeightMode weight_mode, size_t chunk_tokens) {
  if (weight_mode != WeightMode::BFloat16)
    throw std::invalid_argument("Batched prefill requires BF16 weights");
  const std::string chunk_weights = "bf16";
  const std::string chunk_output = "bf16";
  if (chunk_tokens == 4)
    check_artifact(root, chunk_weights + "-prefill-ffn-b4-projection-input",
                   {{"schema_version", 1},
                    {"batch", 4},
                    {"channels", 2048},
                    {"hidden", 8192},
                    {"weights", "bfloat16"},
                    {"activation", "bfloat16"},
                    {"state", "float32"},
                    {"input_layout", "projection_residual_pairs"},
                    {"fp32_arena_floats", 51200},
                    {"bf16_arena_elements", 40960},
                    {"fp32_offsets",
                     {{"shift", 0},
                      {"raw", 2048},
                      {"projected", 34816},
                      {"output", 43008}}},
                    {"bf16_offsets", {{"mixed", 0}, {"activated", 8192}}}});
  if (chunk_tokens == 4)
    for (int stride : {55296, 61440})
      check_artifact(
          root, chunk_output + "-prefill-output-b2-s" + std::to_string(stride),
          {{"shared_program", chunk_output + "-prefill-output-b2-s2048"}});
  for (bool value : {false, true})
    check_artifact(
        root, value ? "prefill-value-recurrence-b2" : "prefill-recurrence-b2",
        {{"schema_version", 1},
         {"batch", 2},
         {"channels", 2048},
         {"head_size", 64},
         {"arena_vectors", value ? 30 : 27},
         {"fused_value", value},
         {"dtype", "float32"},
         {"lanes", value ? 7 : 8},
         {"first_stride", value ? 55296 : 0}});
  for (int stride : {55296, 61440}) {
    const auto name =
        std::string("bf16") + "-prefill-output-b2-s" + std::to_string(stride);
    check_artifact(root, name,
                   {{"schema_version", 1},
                    {"channels", 2048},
                    {"batch", 2},
                    {"cores", 16},
                    {"input_stride", stride},
                    {"dtype", "bfloat16"},
                    {"residual_layout", "separate_tokens"},
                    {"output_layout", "token_projection_residual"}});
  }
  for (int count : {3, 4})
    check_artifact(
        root, "bf16-attention-projections-" + std::to_string(count) + "-b2",
        {{"schema_version", 1},
         {"dtype", "bfloat16"},
         {"channels", 2048},
         {"branches", count},
         {"rank", 256},
         {"cores", 14},
         {"exact_fp32", false},
         {"batch", 2},
         {"arena_stride", count == 4 ? 61440 : 55296},
         {"value_stride", count == 4 ? 61440 : 6144}});
  const char *name = "bf16-prefill-ffn-b2-projection-input";
  check_artifact(
      root, name,
      {{"schema_version", 1},
       {"batch", 2},
       {"channels", 2048},
       {"hidden", 8192},
       {"weights", "bfloat16"},
       {"activation", "bfloat16"},
       {"state", "float32"},
       {"layout", "token_major"},
       {"input_layout", "projection_residual_pairs"},
       {"fp32_arena_floats", 26624},
       {"bf16_arena_elements", 20480},
       {"fp32_offsets",
        {{"shift", 0}, {"raw", 2048}, {"projected", 18432}, {"output", 22528}}},
       {"bf16_offsets", {{"mixed", 0}, {"activated", 4096}}}});
  {
    check_artifact(root, "bf16-prefill-recurrence-projection-b2",
                   {{"schema_version", 1},
                    {"batch", 2},
                    {"channels", 2048},
                    {"head_size", 64},
                    {"arena_vectors", 30},
                    {"lanes", 7},
                    {"cores", 31},
                    {"output_vectors", {25, 26}},
                    {"dtype", "bfloat16"},
                    {"state_dtype", "float32"}});
    const auto fused_ffn = "bf16-prefill-ffn-b2-recurrence-input";
    check_artifact(root, fused_ffn,
                   {{"schema_version", 1},
                    {"batch", 2},
                    {"channels", 2048},
                    {"hidden", 8192},
                    {"weights", "bfloat16"},
                    {"activation", "bfloat16"},
                    {"state", "float32"},
                    {"input_layout", "recurrence_projection_pairs"},
                    {"input_stride", 61440},
                    {"shared_program", "bf16-prefill-ffn-b2-projection-input"},
                    {"fp32_arena_floats", 26624},
                    {"bf16_arena_elements", 20480}});
  }
}
} // namespace rwkv::inference

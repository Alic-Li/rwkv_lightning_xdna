// SPDX-License-Identifier: Apache-2.0
// Offline weight-only error decomposition; does not open an NPU context.
#include "quantization.hpp"
#include "rwkv/io/pth_tensor.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
using namespace rwkv::inference;
using Json = nlohmann::json;
struct Metric {
  double max_abs = 0, square = 0, reference_square = 0;
  size_t count = 0;
  void add(double actual, double expected) {
    double delta = actual - expected;
    max_abs = std::max(max_abs, std::abs(delta));
    square += delta * delta; reference_square += expected * expected; ++count;
  }
  Json json() const {
    return {{"max_abs", max_abs}, {"rmse", std::sqrt(square / count)},
      {"relative_l2", reference_square ? Json(std::sqrt(square / reference_square)) : Json(nullptr)}};
  }
};
int main(int argc, char **argv) {
  try {
    const bool output_only = argc == 3 && std::string(argv[2]) == "--attention-output";
    if (argc != 2 && !output_only)
      throw std::runtime_error("Usage: rwkv-quantization-audit MODEL [--attention-output]");
    Weights weights(argv[1]);
    const std::vector<std::string> suffixes = output_only
        ? std::vector<std::string>{"att.output.weight"}
        : std::vector<std::string>{"ffn.key.weight", "ffn.value.weight"};
    for (size_t layer = 0; layer < weights.layers(); ++layer)
      for (const auto &suffix : suffixes) {
        std::string name = "blocks." + std::to_string(layer) + "." + suffix;
        const auto &tensor = weights.at(name);
        auto rows = quantization::quantize(tensor);
        Metric storage, ideal_int8, scale_effect, total;
        size_t changed_codes = 0, minimum_scale_rows = 0;
        for (size_t row = 0; row < rows.outputs; ++row) {
          float max_abs = 0;
          for (size_t col = 0; col < rows.inputs; ++col)
            max_abs = std::max(max_abs, std::abs(tensor.data[row * rows.inputs + col]));
          float ideal_scale = std::max(max_abs / 127.f, std::numeric_limits<float>::min());
          minimum_scale_rows += rows.scales[row] == 0x1p-24f;
          for (size_t col = 0; col < rows.inputs; ++col) {
            size_t i = row * rows.inputs + col;
            float original = tensor.data[i];
            float baseline = llm_infer::bf16_bits_to_float(llm_infer::float_to_bf16_bits(original));
            int ideal_code = int(std::clamp(std::nearbyint(original / ideal_scale), -127.f, 127.f));
            double ideal = double(ideal_code) * ideal_scale;
            double actual = double(rows.codes[i]) * rows.scales[row];
            storage.add(baseline, original);
            ideal_int8.add(ideal, baseline);
            scale_effect.add(actual, ideal);
            total.add(actual, baseline);
            changed_codes += ideal_code != rows.codes[i];
          }
        }
        std::cout << Json({{"weight", name}, {"shape", tensor.shape},
          {"baseline_bf16_storage_vs_checkpoint", storage.json()},
          {"int8_with_unrounded_scale_vs_bf16", ideal_int8.json()},
          {"fp16_scale_effect_including_code_changes", scale_effect.json()},
          {"actual_int8_vs_bf16", total.json()},
          {"scale_rounding_changed_codes", changed_codes},
          {"code_count", rows.codes.size()}, {"minimum_scale_rows", minimum_scale_rows}}).dump() << std::endl;
      }
  } catch (const std::exception &e) {
    std::cerr << "rwkv-quantization-audit: " << e.what() << '\n';
    return 1;
  }
}

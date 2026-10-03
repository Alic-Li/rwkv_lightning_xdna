// SPDX-License-Identifier: Apache-2.0
#include "device_test_utils.hpp"
#include "quantization.hpp"
#include "weight_layout.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
using namespace rwkv::inference;
using namespace rwkv::xdna;
using namespace rwkv::xdna::test;
namespace q = rwkv::inference::quantization;
using Json = nlohmann::json;
static std::filesystem::path checked_binary(const std::filesystem::path &root, const char *name, bool traced) {
  Json config;
  std::ifstream input(root / name / "config.json");
  input >> config;
  if (bool(config.value("trace_buffer_bytes", size_t(0))) != traced)
    throw std::runtime_error("Trace artifact and TRACE_FILE must be supplied together");
  if (std::string(name) == "int8-channel-mix" &&
      (config.at("tile_bytes") != 4160 || config.at("scale") != "fp16_expanded_fp32"))
    throw std::runtime_error("Incompatible INT8 test weight ABI");
  return root / name / "design.xclbin";
}
struct Stage {
  Session session;
  Guarded input, parameters, weights, diagnostic, output;
  std::unique_ptr<Guarded> trace;
  DeviceRun run;
  std::unique_ptr<Guarded> trace_buffer(const std::filesystem::path &root, const char *name, bool traced) {
    if (!traced) return {};
    Json config;
    std::ifstream input(root / name / "config.json"); input >> config;
    size_t bytes = config.at("trace_buffer_bytes");
    if (!bytes || bytes > 64 * 1024 * 1024 || bytes % 4)
      throw std::runtime_error("Invalid trace buffer size");
    auto result = std::make_unique<Guarded>(session, bytes);
    std::vector<uint32_t> zeros(bytes / 4);
    result->data.upload(zeros.data(), bytes);
    return result;
  }
  DeviceRun prepare() {
    std::vector<DeviceBuffer> args{input.data, parameters.data, weights.data, diagnostic.data, output.data};
    if (trace) args.push_back(trace->data);
    return session.prepare(args);
  }
  void capture(const char *path, const Vector &shift) {
    std::vector<uint32_t> words(trace->bytes / 4);
    trace->data.upload(words.data(), trace->bytes);
    diagnostic.data.upload(shift.data(), shift.size() * 4);
    run.execute();
    trace->data.download(words.data(), trace->bytes);
    guards();
    if (words.back()) throw std::runtime_error("Trace buffer filled");
    while (!words.empty() && words.back() == 0) words.pop_back();
    if (words.empty()) throw std::runtime_error("Empty hardware trace");
    std::ofstream file(path);
    for (auto word : words) file << std::hex << std::setfill('0') << std::setw(8) << word << '\n';
    if (!file) throw std::runtime_error("Cannot write hardware trace");
  }
  Stage(const std::filesystem::path &root, const char *name, size_t bytes, bool traced = false)
      : session(checked_binary(root, name, traced), root / name / "instructions.bin"),
        input(session, 2048 * 4), parameters(session, 6144 * 4),
        weights(session, bytes), diagnostic(session, 22528 * 4), output(session, 4096 * 4),
        trace(trace_buffer(root, name, traced)), run(prepare()) {}
  void guards() {
    for (auto *p : {&input, &parameters, &weights, &diagnostic, &output}) p->guard();
    if (trace) trace->guard();
  }
};
static Json error(const Vector &a, const Vector &b) {
  if (a.size() != b.size()) throw std::runtime_error("Error metric shape mismatch");
  double worst = 0, square = 0, dot = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
      throw std::runtime_error("Nonfinite accuracy vector");
    double e = double(a[i]) - b[i];
    worst = std::max(worst, std::abs(e)); square += e * e;
    dot += double(a[i]) * b[i]; aa += double(a[i]) * a[i]; bb += double(b[i]) * b[i];
  }
  return {{"max_abs", worst}, {"rmse", std::sqrt(square / a.size())},
          {"relative_l2", bb ? Json(std::sqrt(square / bb)) : Json(nullptr)},
          {"cosine", aa && bb ? Json(dot / std::sqrt(aa * bb)) : Json(nullptr)}};
}
static Vector slice(const Vector &v, size_t start, size_t n) {
  return Vector(v.begin() + start, v.begin() + start + n);
}
static Vector oracle(const q::Rows &rows, const Vector &input) {
  Vector rounded(input.size()), out(rows.outputs);
  for (size_t i = 0; i < input.size(); ++i) rounded[i] = expand(bf16(input[i]));
  for (size_t row = 0; row < rows.outputs; ++row) {
    double sum = 0;
    for (size_t col = 0; col < rows.inputs; ++col)
      sum += double(rows.codes[row * rows.inputs + col]) * rounded[col];
    out[row] = float(sum * rows.scales[row]);
  }
  return out;
}
static void close(const Vector &a, const Vector &b, double &worst) {
  if (a.size() != b.size()) throw std::runtime_error("Oracle shape mismatch");
  for (size_t i = 0; i < a.size(); ++i) {
    double e = std::abs(double(a[i]) - b[i]);
    worst = std::max(worst, e);
    // Same stage-oracle tolerance as the existing BF16 ChannelMix test.
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || e > 2e-5 + 2e-5 * std::abs(b[i]))
      throw std::runtime_error("W8A16 oracle mismatch at " + std::to_string(i) + " error=" + std::to_string(e));
  }
}
static Json timing(Stage &stage, const Vector &shift, size_t count) {
  std::vector<double> ms;
  for (size_t i = 0; i < count + 4; ++i) {
    stage.diagnostic.data.upload(shift.data(), shift.size() * 4);
    auto start = std::chrono::steady_clock::now();
    stage.run.execute();
    auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (i >= 4) ms.push_back(elapsed);
  }
  double total = 0;
  for (double x : ms) total += x;
  std::sort(ms.begin(), ms.end());
  return {{"samples", count}, {"mean_ms", total / count},
          {"p50_ms", ms[(count - 1) / 2]}, {"p95_ms", ms[size_t(std::ceil(count * .95)) - 1]}};
}
int main(int argc, char **argv) {
  try {
    if (argc < 3 || argc > 5)
      throw std::runtime_error("Usage: rwkv-channel-mix-int8-test INT8_ROOT BF16_ROOT [ITERATIONS [TRACE_FILE]]");
    size_t count = 200;
    if (argc >= 4) {
      std::string s(argv[3]);
      if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Invalid iteration count");
      count = std::stoull(s);
      if (!count || count > 1000000) throw std::runtime_error("Invalid iteration count");
    }
    std::mt19937 rng(730);
    std::uniform_real_distribution<float> d(-1, 1);
    Tensor key{{8192, 2048}, Vector(8192 * 2048)}, value{{2048, 8192}, Vector(2048 * 8192)};
    for (auto *t : {&key, &value})
      for (float &v : t->data) v = expand(bf16(d(rng) / 64));
    // Zero rows exercise the minimum-scale contract and produce exact zeros.
    std::fill_n(key.data.begin(), 2048, 0.f);
    std::fill_n(value.data.begin(), 8192, 0.f);
    auto qkey = q::quantize(key), qvalue = q::quantize(value);
    auto qw = q::pack(qkey), second = q::pack(qvalue);
    qw.insert(qw.end(), second.begin(), second.end());
    auto bw = weight_layout::to_bf16(weight_layout::channel_mix(key, value));
    Stage quantized(argv[1], "int8-channel-mix", qw.size(), argc == 5);
    Stage baseline(argv[2], "bf16-channel-mix", bw.size() * 2);
    quantized.weights.data.upload(qw.data(), qw.size());
    baseline.weights.data.upload(bw.data(), bw.size() * 2);
    Vector constants(6144), input(2048), shift(2048);
    for (float &v : constants) v = d(rng) / 8;
    for (size_t i = 0; i < 2048; ++i) constants[i] += 1;
    for (float &v : shift) v = d(rng);
    for (auto *s : {&quantized, &baseline}) s->parameters.data.upload(constants.data(), constants.size() * 4);
    Json passes = Json::array();
    double worst = 0;
    for (int pass = 0; pass < 3; ++pass) {
      for (float &v : input) v = d(rng) * (pass + 1);
      Vector qdiag(22528), bdiag(22528), qout(4096), bout(4096);
      for (auto *s : {&quantized, &baseline}) {
        s->input.data.upload(input.data(), input.size() * 4);
        s->diagnostic.data.upload(shift.data(), shift.size() * 4);
        s->run.execute();
      }
      quantized.diagnostic.data.download(qdiag.data(), qdiag.size() * 4);
      baseline.diagnostic.data.download(bdiag.data(), bdiag.size() * 4);
      quantized.output.data.download(qout.data(), qout.size() * 4);
      baseline.output.data.download(bout.data(), bout.size() * 4);
      if (!std::equal(qdiag.begin(), qdiag.begin() + 6144, bdiag.begin()))
        throw std::runtime_error("INT8 changed normalization/shift/mix");
      if (!std::equal(shift.begin(), shift.end(), qdiag.begin() + 2048))
        throw std::runtime_error("W8A16 shift snapshot mismatch");
      auto qraw = slice(qdiag, 6144, 8192), braw = slice(bdiag, 6144, 8192);
      auto qact = slice(qdiag, 14336, 8192), bact = slice(bdiag, 14336, 8192);
      auto key_ref = oracle(qkey, slice(qdiag, 4096, 2048));
      close(qraw, key_ref, worst);
      Vector relu(8192);
      for (size_t i = 0; i < 8192; ++i) { float x = std::max(qraw[i], 0.f); relu[i] = x * x; }
      close(qact, relu, worst);
      auto all_ref = oracle(qvalue, qact), weight_only = oracle(qvalue, bact);
      auto qprojection = slice(qout, 0, 2048), bprojection = slice(bout, 0, 2048);
      close(qprojection, all_ref, worst);
      Vector residual(2048);
      for (size_t i = 0; i < 2048; ++i) residual[i] = qout[i] + input[i];
      close(slice(qout, 2048, 2048), residual, worst);
      passes.push_back({{"pass", pass}, {"key_weight_quantization", error(qraw, braw)},
        {"relu_squared_propagation", error(qact, bact)},
        {"value_weight_only_cpu_oracle", error(weight_only, bprojection)},
        {"key_error_propagated_through_value_cpu_oracle", error(all_ref, weight_only)},
        {"value_device_vs_quantized_oracle", error(qprojection, all_ref)},
        {"value_total_vs_bf16", error(qprojection, bprojection)},
        {"residual_total_vs_bf16", error(slice(qout, 2048, 2048), slice(bout, 2048, 2048))}});
      shift = slice(qdiag, 0, 2048);
    }
    Json timings = Json::array();
    // A/B/B/A, same process and inputs, prebuilt contexts and buffers.
    if (!quantized.trace)
      for (auto *s : {&baseline, &quantized, &quantized, &baseline})
        timings.push_back({{"weights", s == &baseline ? "bf16" : "int8"}, {"latency", timing(*s, shift, count)}});
    for (auto *s : {&quantized, &baseline}) {
      s->guards();
      Vector actual_input(2048), actual_params(6144);
      s->input.data.download(actual_input.data(), actual_input.size() * 4);
      s->parameters.data.download(actual_params.data(), actual_params.size() * 4);
      if (actual_input != input || actual_params != constants)
        throw std::runtime_error("Immutable stage input overwritten");
    }
    std::vector<uint8_t> same(qw.size());
    quantized.weights.data.download(same.data(), same.size());
    if (same != qw) throw std::runtime_error("INT8 weights overwritten");
    if (quantized.trace) quantized.capture(argv[4], shift);
    std::cout << Json({{"trace_capture", bool(quantized.trace)}, {"status", "passed"}, {"scope", "Synthetic ChannelMix; not whole-model accuracy"},
      {"oracle_max_abs", worst}, {"accuracy", passes}, {"timing_abba", timings},
      {"int8_weight_bytes", qw.size()}, {"bf16_weight_bytes", bw.size() * 2}}).dump(2) << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

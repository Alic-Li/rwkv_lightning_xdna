// SPDX-License-Identifier: Apache-2.0
#include "rwkv/inference/graph.hpp"
#include "rwkv/inference/model.hpp"
#include "sampler.h"
#include "tokenizer.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
void usage() {
  std::cout
      << "RWKV-7 BF16 / experimental W8A16 NPU / FP32 CPU reference CLI\n"
         "  rwkv-cli --model MODEL.pth --prompt TEXT [options]\n"
         "  --backend npu|cpu       default npu: all model arithmetic "
         "on NPU\n"
         "  --kernel-dir DIR        default build/kernels/rwkv7-bf16\n"
         "  --weights bf16|int8-ffn|int8-ffn-output  default bf16; INT8 modes are experimental W8A16\n"
         "  --vocab FILE            default assets/rwkv_vocab_v20230424.txt\n"
         "  --prompt-file FILE      UTF-8 prompt file, exclusive with "
         "--prompt\n"
         "  --max-tokens N          default 128; EOS token 0 stops generation\n"
         "  --temperature F         [0.1,5], default 1\n"
         "  --top-k N               1 = greedy, 0 = all, default 40\n"
         "  --top-p F               [0,1], default 0.9\n"
         "  --seed N                default 42\n"
         "  --presence-penalty F    default 0\n"
         "  --frequency-penalty F   default 0\n"
         "  --penalty-decay F       [0,1], default 0.996\n"
         "  --decode graph|resident|eager  default resident on NPU, graph on "
         "CPU; resident keeps "
         "intermediates in NPU BOs\n"
         "  --prefill sequence|decode  default decode; sequence available on "
         "CPU\n"
         "  --threads N             CPU threads, default 8\n"
         "  --tokens 1,2,3 --dump-logits FILE  diagnostic: dump FP32 logits "
         "after every input token\n";
}
long integer(const std::string &s) {
  size_t end = 0;
  long v = std::stol(s, &end);
  if (end != s.size())
    throw std::runtime_error("Invalid integer: " + s);
  return v;
}
float real(const std::string &s) {
  size_t end = 0;
  float v = std::stof(s, &end);
  if (end != s.size() || !std::isfinite(v))
    throw std::runtime_error("Invalid number: " + s);
  return v;
}
} // namespace
int main(int argc, char **argv) {
  const auto process_start = std::chrono::steady_clock::now();
  try {
    std::map<std::string, std::string> opts{
        {"--backend", "npu"},
        {"--weights", "bf16"},
        {"--kernel-dir", "build/kernels/rwkv7-bf16"},
        {"--vocab", "assets/rwkv_vocab_v20230424.txt"},
        {"--max-tokens", "128"},
        {"--temperature", "1"},
        {"--top-k", "40"},
        {"--top-p", "0.9"},
        {"--seed", "42"},
        {"--presence-penalty", "0"},
        {"--frequency-penalty", "0"},
        {"--penalty-decay", "0.996"},
        {"--threads", "8"},
        {"--prefill", "decode"},
        {"--decode", "resident"},
        {"--model", ""},
        {"--prompt", ""},
        {"--prompt-file", ""},
        {"--tokens", ""},
        {"--dump-logits", ""}};
    std::map<std::string, bool> seen;
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (key == "--help" || key == "-h") {
        usage();
        return 0;
      }
      if (!opts.count(key) || i + 1 == argc || seen[key])
        throw std::runtime_error("Unknown, duplicate or incomplete option: " +
                                 key);
      seen[key] = true;
      opts[key] = argv[++i];
    }
    if (opts["--model"].empty())
      throw std::runtime_error("--model is required; use --help");
    int sources = int(seen["--prompt"]) + int(seen["--prompt-file"]) +
                  int(seen["--tokens"]);
    if (sources != 1)
      throw std::runtime_error(
          "Specify exactly one of --prompt, --prompt-file or --tokens");
    const long max_tokens = integer(opts["--max-tokens"]),
               topk = integer(opts["--top-k"]),
               threads = integer(opts["--threads"]),
               seed = integer(opts["--seed"]);
    const float temperature = real(opts["--temperature"]),
                topp = real(opts["--top-p"]),
                presence = real(opts["--presence-penalty"]),
                frequency = real(opts["--frequency-penalty"]),
                decay = real(opts["--penalty-decay"]);
    if (max_tokens < 0 || max_tokens > 1000000 || topk < 0 ||
        topk > 2147483647 || threads < 1 || threads > 256 ||
        seed < -2147483648L || seed > 2147483647L || temperature < 0.1f ||
        temperature > 5 || topp < 0 || topp > 1 || presence < 0 ||
        frequency < 0 || decay < 0 || decay > 1)
      throw std::runtime_error("Sampling/generation option out of range");
    if (opts["--backend"] != "cpu" && opts["--backend"] != "npu")
      throw std::runtime_error("--backend must be cpu or npu");
    if (opts["--weights"] != "bf16" && opts["--weights"] != "int8-ffn" &&
        opts["--weights"] != "int8-ffn-output")
      throw std::runtime_error("--weights must be bf16, int8-ffn or int8-ffn-output");
    if (opts["--weights"] != "bf16" && opts["--backend"] != "npu")
      throw std::runtime_error("INT8 weights require --backend npu");
#ifdef _OPENMP
    omp_set_num_threads(int(threads));
#endif
    std::vector<int> tokens;
    trie_tokenizer tokenizer;
    const bool diagnostic = seen["--tokens"];
    if (diagnostic) {
      std::istringstream in(opts["--tokens"]);
      std::string s;
      while (std::getline(in, s, ',')) {
        long id = integer(s);
        if (id < 0 || id > 2147483647)
          throw std::runtime_error("Invalid token id");
        tokens.push_back(int(id));
      }
      if (tokens.empty())
        throw std::runtime_error("Empty token sequence");
    } else {
      if (tokenizer.load(opts["--vocab"]) != RWKV_SUCCESS)
        throw std::runtime_error("Cannot load vocabulary");
      std::string prompt = opts["--prompt"];
      if (seen["--prompt-file"]) {
        std::ifstream in(opts["--prompt-file"], std::ios::binary);
        if (!in)
          throw std::runtime_error("Cannot open prompt file");
        prompt.assign(std::istreambuf_iterator<char>(in), {});
      }
      if (prompt.empty())
        throw std::runtime_error("Prompt must not be empty");
      tokens = tokenizer.encode(prompt);
      if (tokens.empty() || tokenizer.decode(tokens) != prompt)
        throw std::runtime_error("Tokenizer did not round-trip entire prompt");
    }
    std::cerr << "Loading " << opts["--model"] << " (FP32 host weights)...\n";
    rwkv::inference::Weights weights(opts["--model"]);
    for (int token : tokens)
      if (size_t(token) >= weights.vocabulary())
        throw std::runtime_error("Prompt token exceeds model vocabulary");
    std::cerr << "RWKV-7: " << weights.layers() << " layers, "
              << weights.channels() << " channels, " << weights.heads()
              << " heads; backend=" << opts["--backend"]
              << "; weights=" << (opts["--backend"] == "npu" ? opts["--weights"] : "fp32_reference")
              << (opts["--backend"] == "npu"
                      ? " (all model arithmetic on NPU; CPU embedding/sampling)"
                      : " (all CPU)")
              << '\n';
    // The CPU backend is used only for the explicit CPU reference and state
    // creation. NPU graph construction has no arithmetic backend to fall back
    // to.
    auto backend = rwkv::inference::cpu_backend();
    if (opts["--backend"] == "cpu" && !seen["--decode"])
      opts["--decode"] = "graph";
    if (opts["--backend"] == "npu" &&
        (opts["--decode"] != "resident" || opts["--prefill"] != "decode"))
      throw std::runtime_error(
          "NPU inference requires --decode resident --prefill decode");
    rwkv::inference::Model model(weights, *backend);
    auto state = model.initial_state();
    if (opts["--decode"] != "graph" && opts["--decode"] != "resident" &&
        opts["--decode"] != "eager")
      throw std::runtime_error("--decode must be graph, resident or eager");
    if (opts["--decode"] == "resident" && opts["--backend"] != "npu")
      throw std::runtime_error("resident decode requires --backend npu");
    std::unique_ptr<rwkv::inference::DecodeGraph> graph;
    double graph_setup_seconds = 0;
    auto prepare_graph = [&]() {
      if (graph || opts["--decode"] == "eager")
        return;
      const auto graph_start = std::chrono::steady_clock::now();
      if (opts["--decode"] == "resident" && opts["--backend"] != "npu")
        throw std::runtime_error("resident decode requires --backend npu");
      if (opts["--backend"] == "npu")
        graph = std::make_unique<rwkv::inference::DecodeGraph>(
            weights, opts["--kernel-dir"], opts["--weights"] == "int8-ffn-output"
                ? rwkv::inference::WeightMode::Int8FFNOutput
                : opts["--weights"] == "int8-ffn"
                ? rwkv::inference::WeightMode::Int8FFN
                : rwkv::inference::WeightMode::BFloat16);
      else
        graph =
            std::make_unique<rwkv::inference::DecodeGraph>(weights, *backend);
      graph_setup_seconds += std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - graph_start)
                                 .count();
      auto stats = graph->stats();
      if (stats.device_runs)
        std::cerr << "Resident decode: " << stats.persistent_runs
                  << " runs/token, " << stats.resident_bytes << " BO bytes in "
                  << stats.root_bos << " root BOs; persistent upload/download "
                  << stats.persistent_upload_bytes << "/"
                  << stats.persistent_download_bytes << " bytes\n";
      std::cerr << "Captured decode graph: " << stats.nodes << " nodes, "
                << stats.buffers << " buffers (native run submissions)\n";
    };
    bool state_on_device = false;
    auto decode = [&](int token) {
      prepare_graph();
      if (opts["--decode"] == "resident") {
        if (!state_on_device) {
          graph->load_state(state);
          state_on_device = true;
        }
        return graph->replay_resident(token);
      }
      return graph ? graph->replay(token, state) : model.forward(token, state);
    };
    std::ofstream dump;
    if (!opts["--dump-logits"].empty()) {
      dump.open(opts["--dump-logits"], std::ios::binary);
      if (!dump)
        throw std::runtime_error("Cannot open logits output");
    }
    auto start = std::chrono::steady_clock::now();
    const double setup_before_prefill = graph_setup_seconds;
    rwkv::inference::Vector logits;
    if (opts["--prefill"] != "sequence" && opts["--prefill"] != "decode")
      throw std::runtime_error("--prefill must be sequence or decode");
    auto save_logits = [&](const rwkv::inference::Vector &row) {
      if (dump.is_open()) {
        dump.write(reinterpret_cast<const char *>(row.data()),
                   row.size() * sizeof(float));
        if (!dump)
          throw std::runtime_error("Logits write failed");
      }
    };
    if (opts["--prefill"] == "sequence") {
      for (size_t start = 0; start < tokens.size(); start += 16) {
        const size_t end = std::min(tokens.size(), start + 16);
        auto rows = model.prefill(
            std::vector<int>(tokens.begin() + start, tokens.begin() + end),
            state);
        for (const auto &row : rows)
          save_logits(row);
        logits = std::move(rows.back());
      }
    } else {
      for (int token : tokens) {
        logits = decode(token);
        save_logits(logits);
      }
    }
    auto prefilled = std::chrono::steady_clock::now();
    std::cerr << "Prefill " << tokens.size() << " tokens: "
              << std::chrono::duration<double>(prefilled - start).count()
              << " s\n";
    std::cerr << "Prefill excluding graph setup: "
              << std::chrono::duration<double>(prefilled - start).count() -
                     (graph_setup_seconds - setup_before_prefill)
              << " s; graph setup: " << graph_setup_seconds << " s\n";
    if (diagnostic)
      return 0;
    auto setup_start = std::chrono::steady_clock::now();
    prepare_graph();
    if (opts["--decode"] == "resident" && !state_on_device) {
      graph->load_state(state);
      state_on_device = true;
    }
    auto generation_start = std::chrono::steady_clock::now();
    std::cerr
        << "Decode setup after prefill: "
        << std::chrono::duration<double>(generation_start - setup_start).count()
        << " s\n";
    rwkvmobile::NucleusSampler sampler;
    sampler.set_seed(int32_t(seed));
    std::map<int, float> occurrences;
    size_t generated = 0, decode_calls = 0;
    double decode_seconds = 0;
    for (long i = 0; i < max_tokens; ++i) {
      rwkvmobile::Tensor1D view{logits.data(), rwkvmobile::TensorDType::F32,
                                logits.size()};
      sampler.apply_penalties(view, logits.size(), occurrences, {}, presence,
                              frequency, decay);
      int token =
          sampler.sample(view, logits.size(), temperature, int(topk), topp);
      if (token == tokenizer.eos_token_id)
        break;
      std::string piece = tokenizer.decode(token);
      if (piece.empty())
        throw std::runtime_error("Generated token missing from vocabulary: " +
                                 std::to_string(token));
      if (!generated) {
        const auto first_token = std::chrono::steady_clock::now();
        std::cerr << "TTFT from process entry: "
                  << std::chrono::duration<double>(first_token - process_start)
                         .count()
                  << " s; from prefill start: "
                  << std::chrono::duration<double>(first_token - start).count()
                  << " s\n";
      }
      std::cout << piece << std::flush;
      ++generated;
      occurrences[token] += 1;
      if (i + 1 < max_tokens) {
        const auto decode_start = std::chrono::steady_clock::now();
        logits = decode(token);
        decode_seconds += std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - decode_start)
                              .count();
        ++decode_calls;
      }
    }
    std::cout << '\n';
    std::cerr << "Generated " << generated << " tokens in "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - generation_start)
                     .count()
              << " s\n";
    std::cerr << "Generation decode forwards: " << decode_calls << " in "
              << decode_seconds << " s; "
              << (decode_calls ? decode_calls / decode_seconds : 0)
              << " token/s\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "rwkv-cli: " << e.what() << '\n';
    return 1;
  }
}

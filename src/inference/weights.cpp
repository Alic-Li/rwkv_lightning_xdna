// SPDX-License-Identifier: Apache-2.0
#include "rwkv/inference/model.hpp"
#include "rwkv/io/pth_tensor.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace rwkv::inference {
namespace {
float half(uint16_t x) {
  const int e = (x >> 10) & 31, m = x & 1023;
  float v = e == 0    ? std::ldexp(float(m), -24)
            : e == 31 ? (m ? NAN : INFINITY)
                      : std::ldexp(float(1024 + m), e - 25);
  return x & 32768 ? -v : v;
}
} // namespace
Weights::Weights(const std::filesystem::path &path) {
  if (path.extension() == ".pth" || path.extension() == ".pt") {
    auto archive = llm_infer::PthArchive::open(path.string(), true);
    if (!archive.ok())
      throw std::runtime_error(archive.status().message());
    for (const auto &entry : archive.value().entries()) {
      if (entry.name.size() >= 10 &&
          entry.name.substr(entry.name.size() - 10) == "/byteorder") {
        auto bytes = archive.value().read_stored_entry(entry);
        if (!bytes.ok())
          throw std::runtime_error(bytes.status().message());
        if (std::string(bytes.value().begin(), bytes.value().end()) != "little")
          throw std::runtime_error(
              "Only little-endian PTH storage is supported");
      }
    }
    auto records = llm_infer::parse_pth_tensor_records(archive.value());
    if (!records.ok())
      throw std::runtime_error(records.status().message());
    for (const auto &record : records.value()) {
      // Reject malformed strides/ranges before entering the upstream converter.
      if (record.shape.size() != record.stride.size())
        throw std::runtime_error("PTH shape/stride mismatch");
      uint64_t max_offset = record.storage_offset;
      size_t count = 1;
      for (size_t i = 0; i < record.shape.size(); ++i) {
        if (record.shape[i] <= 0 || record.stride[i] < 0 ||
            uint64_t(record.shape[i]) >
                std::numeric_limits<size_t>::max() / count)
          throw std::runtime_error("Invalid PTH tensor shape/stride");
        count *= size_t(record.shape[i]);
        const uint64_t extent = uint64_t(record.shape[i] - 1),
                       stride = uint64_t(record.stride[i]);
        if (stride &&
            extent >
                (std::numeric_limits<uint64_t>::max() - max_offset) / stride)
          throw std::runtime_error("PTH stride overflow");
        max_offset += extent * stride;
      }
      if (max_offset >= record.storage_size ||
          record.storage_size > std::numeric_limits<uint64_t>::max() / 4)
        throw std::runtime_error("PTH tensor outside storage");
      Tensor t;
      t.shape.assign(record.shape.begin(), record.shape.end());
      size_t stride = 1;
      bool contiguous = true;
      for (size_t i = record.shape.size(); i-- > 0;) {
        if (record.shape[i] > 1 && size_t(record.stride[i]) != stride)
          contiguous = false;
        stride *= size_t(record.shape[i]);
      }
      if (contiguous) {
        // Decode contiguous storage in a single typed loop, avoiding
        // per-element conversion to unused BF16/F16 copies in the CUDA
        // convenience loader.
        std::string suffix = "/data/" + record.storage_key;
        const llm_infer::PthEntry *entry = nullptr;
        for (const auto &candidate : archive.value().entries())
          if (candidate.name.size() >= suffix.size() &&
              candidate.name.compare(candidate.name.size() - suffix.size(),
                                     suffix.size(), suffix) == 0) {
            if (entry)
              throw std::runtime_error("Ambiguous PTH storage entry");
            entry = &candidate;
          }
        if (!entry)
          throw std::runtime_error("Missing PTH storage");
        const size_t width = llm_infer::dtype_size_bytes(record.dtype);
        if (entry->uncompressed_size != record.storage_size * width)
          throw std::runtime_error("PTH storage length mismatch");
        std::vector<uint8_t> raw(count * width);
        auto status = archive.value().read_stored_entry_range(
            *entry, record.storage_offset * width, raw.data(), raw.size());
        if (!status.ok_status())
          throw std::runtime_error(status.message());
        t.data.resize(count);
        if (record.dtype == llm_infer::TensorDType::kFloat32)
          std::memcpy(t.data.data(), raw.data(), raw.size());
        else if (record.dtype == llm_infer::TensorDType::kBFloat16) {
          for (size_t i = 0; i < count; ++i) {
            uint16_t bits;
            std::memcpy(&bits, raw.data() + 2 * i, 2);
            uint32_t f = uint32_t(bits) << 16;
            std::memcpy(t.data.data() + i, &f, 4);
          }
        } else {
          for (size_t i = 0; i < count; ++i) {
            uint16_t bits;
            std::memcpy(&bits, raw.data() + 2 * i, 2);
            t.data[i] = half(bits);
          }
        }
      } else {
        auto loaded = llm_infer::load_tensor_select(archive.value(), record,
                                                    false, false, true);
        if (!loaded.ok())
          throw std::runtime_error(loaded.status().message());
        t.data = std::move(loaded.value().values);
      }
      if (t.data.size() != count)
        throw std::runtime_error("PTH tensor size mismatch");
      for (float v : t.data)
        if (!std::isfinite(v))
          throw std::runtime_error("Nonfinite weight: " + record.name);
      if (!tensors_.emplace(record.name, std::move(t)).second)
        throw std::runtime_error("Duplicate PTH tensor");
    }
  } else {
    std::ifstream in(path, std::ios::binary);
    if (!in)
      throw std::runtime_error("Cannot open weights: " + path.string());
    const auto length = std::filesystem::file_size(path);
    uint64_t header_size = 0;
    in.read(reinterpret_cast<char *>(&header_size), 8);
    if (!in || length < 8 || header_size > length - 8 ||
        header_size > 100000000)
      throw std::runtime_error(
          "Invalid safetensors header (convert .pth offline first)");
    std::string header(header_size, '\0');
    in.read(header.data(), header.size());
    auto json = nlohmann::json::parse(header);
    std::vector<std::pair<uint64_t, uint64_t>> spans;
    for (auto it = json.begin(); it != json.end(); ++it) {
      if (it.key() == "__metadata__")
        continue;
      const auto &j = it.value();
      Tensor t;
      t.shape = j.at("shape").get<std::vector<size_t>>();
      size_t count = 1;
      for (size_t d : t.shape) {
        if (!d || count > std::numeric_limits<size_t>::max() / d)
          throw std::runtime_error("Invalid tensor shape: " + it.key());
        count *= d;
      }
      const auto dtype = j.at("dtype").get<std::string>();
      const size_t width = dtype == "F32"                        ? 4
                           : (dtype == "F16" || dtype == "BF16") ? 2
                                                                 : 0;
      if (!width)
        throw std::runtime_error("Unsupported dtype " + dtype + ": " +
                                 it.key());
      const auto offsets = j.at("data_offsets").get<std::vector<uint64_t>>();
      if (offsets.size() != 2 || offsets[1] < offsets[0] ||
          offsets[1] > length - 8 - header_size ||
          count > std::numeric_limits<size_t>::max() / width ||
          offsets[1] - offsets[0] != count * width)
        throw std::runtime_error("Invalid tensor offsets: " + it.key());
      spans.emplace_back(offsets[0], offsets[1]);
      in.seekg(8 + header_size + offsets[0]);
      std::vector<uint8_t> raw(count * width);
      in.read(reinterpret_cast<char *>(raw.data()), raw.size());
      if (!in)
        throw std::runtime_error("Truncated tensor: " + it.key());
      t.data.resize(count);
      for (size_t i = 0; i < count; ++i) {
        if (dtype == "F32")
          std::memcpy(&t.data[i], raw.data() + 4 * i, 4);
        else {
          uint16_t bits;
          std::memcpy(&bits, raw.data() + 2 * i, 2);
          if (dtype == "F16")
            t.data[i] = half(bits);
          else {
            uint32_t f = uint32_t(bits) << 16;
            std::memcpy(&t.data[i], &f, 4);
          }
        }
        if (!std::isfinite(t.data[i]))
          throw std::runtime_error("Nonfinite weight: " + it.key());
      }
      tensors_.emplace(it.key(), std::move(t));
    }
    std::sort(spans.begin(), spans.end());
    uint64_t end = 0;
    for (auto [begin, next] : spans) {
      if (begin != end)
        throw std::runtime_error(
            "Noncontiguous or overlapping safetensors payload");
      end = next;
    }
    if (end != length - header_size - 8)
      throw std::runtime_error("Unexpected safetensors trailing data");
  }
  const auto &emb = at("emb.weight");
  if (emb.shape.size() != 2)
    throw std::runtime_error("emb.weight must be [vocab, channels]");
  vocabulary_ = emb.shape[0];
  channels_ = emb.shape[1];
  while (tensors_.count("blocks." + std::to_string(layers_) + ".ln1.weight"))
    ++layers_;
  if (!layers_)
    throw std::runtime_error("No RWKV-7 layers found");
  const auto &rk = at("blocks.0.att.r_k");
  if (rk.shape.size() != 2 || rk.data.size() != channels_)
    throw std::runtime_error("att.r_k must be [heads, head_size]");
  heads_ = rk.shape[0];
  auto vector = [&](const std::string &n) {
    const auto &t = at(n);
    if (t.data.size() != channels_ ||
        std::count_if(t.shape.begin(), t.shape.end(),
                      [](size_t d) { return d != 1; }) > 1)
      throw std::runtime_error("Invalid channel vector: " + n);
  };
  auto matrix = [&](const std::string &n, size_t rows, size_t cols) {
    if (at(n).shape != std::vector<size_t>{rows, cols})
      throw std::runtime_error("Invalid matrix shape: " + n);
  };
  vector("blocks.0.ln0.weight");
  vector("blocks.0.ln0.bias");
  vector("ln_out.weight");
  vector("ln_out.bias");
  matrix("head.weight", vocabulary_, channels_);
  for (size_t l = 0; l < layers_; ++l) {
    const std::string p = "blocks." + std::to_string(l) + ".";
    for (auto n : {"ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
                   "att.x_r", "att.x_w", "att.x_k", "att.x_v", "att.x_a",
                   "att.x_g", "att.w0", "att.a0", "att.k_k", "att.k_a",
                   "att.ln_x.weight", "att.ln_x.bias", "ffn.x_k"})
      vector(p + n);
    matrix(p + "att.r_k", heads_, head_size());
    for (auto n : {"att.receptance.weight", "att.key.weight",
                   "att.value.weight", "att.output.weight"})
      matrix(p + n, channels_, channels_);
    for (auto n : {"w", "a", "g", "v"}) {
      if (l == 0 && std::string(n) == "v")
        continue;
      const auto base = p + "att." + n;
      const auto &shape = at(base + "1").shape;
      if (shape.size() != 2 || shape[0] != channels_)
        throw std::runtime_error("Invalid low-rank input: " + base);
      matrix(base + "2", shape[1], channels_);
      if (std::string(n) == "v")
        vector(base + "0");
    }
    const auto &fs = at(p + "ffn.key.weight").shape;
    if (fs.size() != 2 || fs[1] != channels_)
      throw std::runtime_error("Invalid FFN shape");
    matrix(p + "ffn.value.weight", channels_, fs[0]);
  }
}
const Tensor &Weights::at(const std::string &n) const {
  auto it = tensors_.find(n);
  if (it == tensors_.end())
    throw std::runtime_error("Missing RWKV-7 tensor: " + n);
  return it->second;
}
} // namespace rwkv::inference

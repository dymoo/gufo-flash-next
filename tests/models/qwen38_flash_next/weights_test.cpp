// Binding K-quant dense projections: OpenBMB's MiniCPM-V 4.7 Q4_K_M stores
// them as Q4_K/Q5_K/Q6_K. A sparse two-layer qwen35moe file (headers only;
// payloads are holes) is enough, so no model or GPU is needed.
#include "src/models/qwen38_flash_next/weights.hpp"

#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qfn = gufo::models::qwen38_flash_next;
using gufo::core::GgmlType;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct Tensor {
  std::string name;
  std::vector<std::uint64_t> dims;
  GgmlType type;
};

std::uint64_t Bytes(const Tensor& t) {
  std::uint64_t n = 1;
  for (auto d : t.dims)
    n *= d;
  switch (t.type) {
    case GgmlType::kF32:
      return n * 4;
    case GgmlType::kQ4_0:
      return n / 32 * 18;
    case GgmlType::kQ4_K:
      return n / 256 * 144;
    case GgmlType::kQ5_K:
      return n / 256 * 176;
    case GgmlType::kQ6_K:
      return n / 256 * 210;
    default:
      throw std::runtime_error("test tensor type");
  }
}

/// Layers 2 (Gated DeltaNet) and 3 (gated attention) of the official file,
/// with its per-tensor K-quant mix, plus the embedding, head and norm.
std::vector<Tensor> OfficialMix() {
  constexpr std::uint64_t vocab = 256;
  const auto q4 = GgmlType::kQ4_K, q5 = GgmlType::kQ5_K, q6 = GgmlType::kQ6_K,
             f32 = GgmlType::kF32;
  std::vector<Tensor> t = {{"token_embd.weight", {2048, vocab}, q4},
                           {"output.weight", {2048, vocab}, q6},
                           {"output_norm.weight", {2048}, f32}};
  for (int il : {2, 3}) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::vector<Tensor> common = {
        {p + "attn_norm.weight", {2048}, f32},
        {p + "post_attention_norm.weight", {2048}, f32},
        {p + "ffn_gate_inp.weight", {2048, 256}, f32},
        {p + "ffn_gate_exps.weight", {2048, 512, 256}, q4},
        {p + "ffn_up_exps.weight", {2048, 512, 256}, q4},
        {p + "ffn_down_exps.weight", {512, 2048, 256}, il == 2 ? q6 : q4},
        {p + "ffn_gate_inp_shexp.weight", {2048}, f32},
        {p + "ffn_gate_shexp.weight", {2048, 512}, q4},
        {p + "ffn_up_shexp.weight", {2048, 512}, q4},
        {p + "ffn_down_shexp.weight", {512, 2048}, q6}};
    t.insert(t.end(), common.begin(), common.end());
  }
  const std::vector<Tensor> gdn = {{"blk.2.attn_qkv.weight", {2048, 8192}, q6},
                                   {"blk.2.attn_gate.weight", {2048, 4096}, q4},
                                   {"blk.2.ssm_conv1d.weight", {4, 8192}, f32},
                                   {"blk.2.ssm_alpha.weight", {2048, 32}, q4},
                                   {"blk.2.ssm_beta.weight", {2048, 32}, q4},
                                   {"blk.2.ssm_dt.bias", {32}, f32},
                                   {"blk.2.ssm_a", {32}, f32},
                                   {"blk.2.ssm_norm.weight", {128}, f32},
                                   {"blk.2.ssm_out.weight", {4096, 2048}, q5}};
  const std::vector<Tensor> attention = {
      {"blk.3.attn_q.weight", {2048, 8192}, q4},
      {"blk.3.attn_k.weight", {2048, 512}, q4},
      {"blk.3.attn_v.weight", {2048, 512}, q6},
      {"blk.3.attn_output.weight", {4096, 2048}, q4},
      {"blk.3.attn_q_norm.weight", {256}, f32},
      {"blk.3.attn_k_norm.weight", {256}, f32}};
  t.insert(t.end(), gdn.begin(), gdn.end());
  t.insert(t.end(), attention.begin(), attention.end());
  return t;
}

/// Writes the qwen35moe metadata and tensor headers, then extends the file
/// sparsely past every payload.
std::filesystem::path Write(const std::vector<Tensor>& tensors,
                            const std::string& name) {
  std::vector<std::pair<std::string, std::uint32_t>> u32 = {
      {"block_count", 40},
      {"embedding_length", 2048},
      {"context_length", 262144},
      {"full_attention_interval", 4},
      {"attention.head_count", 16},
      {"attention.head_count_kv", 2},
      {"attention.key_length", 256},
      {"rope.dimension_count", 64},
      {"ssm.conv_kernel", 4},
      {"ssm.state_size", 128},
      {"ssm.group_count", 16},
      {"ssm.time_step_rank", 32},
      {"ssm.inner_size", 4096},
      {"expert_count", 256},
      {"expert_used_count", 8},
      {"expert_feed_forward_length", 512},
      {"expert_shared_feed_forward_length", 512}};
  std::vector<std::uint8_t> b;
  const auto pod = [&]<class T>(T v) {
    const auto at = b.size();
    b.resize(at + sizeof(v));
    std::memcpy(b.data() + at, &v, sizeof(v));
  };
  const auto text = [&](const std::string& s) {
    pod(std::uint64_t{s.size()});
    b.insert(b.end(), s.begin(), s.end());
  };
  pod(std::uint32_t{0x46554747});
  pod(std::uint32_t{3});
  pod(std::uint64_t{tensors.size()});
  pod(std::uint64_t{u32.size() + 4});
  text("general.architecture");
  pod(std::uint32_t{8});
  text("qwen35moe");
  for (const auto& [key, value] : u32) {
    text("qwen35moe." + key);
    pod(std::uint32_t{4});
    pod(value);
  }
  for (const auto& [key, value] :
       {std::pair{"attention.layer_norm_rms_epsilon", 1e-6F},
        std::pair{"rope.freq_base", 1e7F}}) {
    text(std::string("qwen35moe.") + key);
    pod(std::uint32_t{6});
    pod(value);
  }
  text("qwen35moe.rope.dimension_sections");
  pod(std::uint32_t{9});
  pod(std::uint32_t{10});
  pod(std::uint64_t{4});
  for (std::uint64_t v : {11, 11, 10, 0})
    pod(v);
  std::uint64_t offset = 0;
  for (const auto& t : tensors) {
    text(t.name);
    pod(static_cast<std::uint32_t>(t.dims.size()));
    for (auto d : t.dims)
      pod(d);
    pod(static_cast<std::uint32_t>(t.type));
    pod(offset);
    offset += (Bytes(t) + 31) / 32 * 32;
  }
  b.resize((b.size() + 31) / 32 * 32);
  const auto path = std::filesystem::temp_directory_path() /
                    ("gufo-kquant-" + std::to_string(::getpid()) + name);
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(b.data()),
              static_cast<std::streamsize>(b.size()));
  }
  std::filesystem::resize_file(path, b.size() + offset);
  return path;
}

std::optional<qfn::ModelWeights> Bind(const std::vector<Tensor>& tensors,
                                      std::string* error) {
  const auto path = Write(tensors, ".gguf");
  const auto reader = gufo::core::GgufReader::OpenFile(path, error);
  std::filesystem::remove(path);
  Require(reader != nullptr, "sparse GGUF did not open: " + *error);
  return qfn::ModelWeights::Bind(*reader, error, 2, 2);
}

}  // namespace

int main() {
  try {
    std::string error;
    const auto w = Bind(OfficialMix(), &error);
    Require(w.has_value(), "official K-quant mix rejected: " + error);
    Require(w->token_embd.type == GgmlType::kQ4_K &&
                w->output.type == GgmlType::kQ6_K && w->layers.size() == 2,
            "embedding/head binding");
    const auto& gdn = w->layers[0];
    const auto& attention = w->layers[1];
    Require(gdn.linear() && !attention.linear(), "layer slice kinds");
    Require(gdn.ssm_qkv.type == GgmlType::kQ6_K &&
                gdn.ssm_gate.type == GgmlType::kQ4_K &&
                gdn.ssm_alpha.type == GgmlType::kQ4_K &&
                gdn.ssm_out.type == GgmlType::kQ5_K &&
                gdn.ffn_down_exps.type == GgmlType::kQ6_K &&
                attention.attn_v.type == GgmlType::kQ6_K &&
                attention.shexp_gate.type == GgmlType::kQ4_K,
            "per-tensor K-quant types");
    Require(gdn.ssm_out.RowBytes() == 4096 / 256 * 176 &&
                gdn.ssm_qkv.SizeBytes() == std::size_t{8192} * 8 * 210 &&
                attention.shexp_down.RowBytes() == 2 * 210 &&
                w->output.SizeBytes() == std::size_t{256} * 8 * 210,
            "K-quant row and tensor sizes");
    // A dense format the engine has no kernel for still fails at binding,
    // naming the tensor.
    auto wrong = OfficialMix();
    for (auto& t : wrong)
      if (t.name == "blk.2.attn_gate.weight")
        t.type = GgmlType::kQ4_0;
    Require(!Bind(wrong, &error).has_value() &&
                error.find("blk.2.attn_gate.weight") != std::string::npos &&
                error.find("unsupported format") != std::string::npos,
            "unsupported dense format accepted: " + error);
    std::cout << "Flash-Next K-quant binding checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

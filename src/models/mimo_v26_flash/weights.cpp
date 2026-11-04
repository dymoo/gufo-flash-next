#include "src/models/mimo_v26_flash/weights.hpp"

#include <initializer_list>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::mimo_v26_flash {
namespace {

using core::GgmlType;

/// Expert formats of the imatrix IQ2_M-class mix; every one has a kernel.
constexpr std::initializer_list<GgmlType> kExpertTypes = {
    GgmlType::kIQ2_XXS, GgmlType::kIQ2_XS, GgmlType::kIQ2_S, GgmlType::kIQ3_XXS,
    GgmlType::kIQ3_S,   GgmlType::kQ3_K,   GgmlType::kMXFP4};
constexpr std::initializer_list<GgmlType> kProjectionTypes = {GgmlType::kQ6_K,
                                                              GgmlType::kQ8_0};

struct Binder {
  const core::GgufReader& reader;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr)
      *error = std::move(message);
    ok = false;
  }

  TensorRef Get(const std::string& name, std::uint64_t cols, std::uint64_t rows,
                std::uint64_t experts, std::initializer_list<GgmlType> types,
                bool required = true) {
    TensorRef t;
    const auto* info = reader.FindTensor(name);
    if (info == nullptr) {
      if (required)
        Fail("missing tensor " + name);
      return t;
    }
    const auto& d = info->dimensions;
    const std::uint64_t got_rows = d.size() > 1 ? d[1] : 1;
    const std::uint64_t got_experts = d.size() > 2 ? d[2] : 1;
    if (d.empty() || d[0] != cols || got_rows != rows ||
        got_experts != experts) {
      Fail("tensor " + name + " has an unexpected shape");
      return t;
    }
    bool type_ok = false;
    for (const auto type : types)
      type_ok = type_ok || info->type == type;
    if (!type_ok) {
      Fail("tensor " + name + " has unsupported format " +
           std::string(core::ToString(info->type)));
      return t;
    }
    t.data = info->data;
    t.type = info->type;
    t.cols = cols;
    t.rows = rows;
    t.experts = experts;
    t.name = info->name;
    const auto regions = reader.GetMappedRegions();
    const auto address = reinterpret_cast<std::uintptr_t>(info->data);
    for (std::uint32_t i = 0; i < regions.size(); ++i) {
      const auto base = reinterpret_cast<std::uintptr_t>(regions[i].data);
      if (address >= base && address < base + regions[i].size) {
        t.shard = i;
        t.file_offset = address - base;
        if (t.SizeBytes() > regions[i].size - t.file_offset)
          Fail("tensor " + name + " is truncated");
        return t;
      }
    }
    Fail("tensor " + name + " lies outside every shard");
    return TensorRef{};
  }
};

}  // namespace

std::size_t TensorRef::RowBytes() const noexcept {
  return quant::EncodedSizeBytes(type, cols);
}

std::size_t ModelWeights::TrunkBytes() const noexcept {
  std::size_t bytes =
      token_embd.SizeBytes() + output_norm.SizeBytes() + output.SizeBytes();
  for (const auto& l : layers)
    for (const auto* t :
         {&l.attn_norm, &l.attn_qkv, &l.attn_out, &l.attn_sinks, &l.ffn_norm,
          &l.ffn_gate, &l.ffn_up, &l.ffn_down, &l.router, &l.router_bias,
          &l.gate_exps, &l.up_exps, &l.down_exps})
      bytes += t->empty() ? 0 : t->SizeBytes();
  return bytes;
}

std::optional<ModelWeights> ModelWeights::Bind(const core::GgufReader& reader,
                                               std::string* error_msg) {
  auto config = Config::Parse(reader, error_msg);
  if (!config)
    return std::nullopt;
  const auto& c = *config;
  constexpr auto H = Config::kHidden;
  Binder b{reader, error_msg};
  ModelWeights w;
  w.config = c;
  w.token_embd = b.Get("token_embd.weight", H, c.vocab, 1, {GgmlType::kQ8_0});
  w.output_norm = b.Get("output_norm.weight", H, 1, 1, {GgmlType::kF32});
  w.output = b.Get("output.weight", H, c.vocab, 1, kProjectionTypes);
  w.layers.resize(Config::kLayers);
  for (std::uint32_t l = 0; l < Config::kLayers && b.ok; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    auto& L = w.layers[l];
    L.attn_norm = b.Get(p + "attn_norm.weight", H, 1, 1, {GgmlType::kF32});
    L.attn_qkv =
        b.Get(p + "attn_qkv.weight", H, c.QkvRows(l), 1, kProjectionTypes);
    L.attn_out = b.Get(p + "attn_output.weight", Config::kAttnOutDim, H, 1,
                       kProjectionTypes);
    L.attn_sinks = b.Get(p + "attn_sinks.weight", Config::kHeads, 1, 1,
                         {GgmlType::kF32}, c.swa[l]);
    if (c.swa[l] == L.attn_sinks.empty()) {
      b.Fail("attention sinks must exist exactly on sliding-window layers");
      break;
    }
    L.ffn_norm = b.Get(p + "ffn_norm.weight", H, 1, 1, {GgmlType::kF32});
    if (c.dense_ffn[l]) {
      L.ffn_gate = b.Get(p + "ffn_gate.weight", H, Config::kDenseFf, 1,
                         {GgmlType::kQ8_0});
      L.ffn_up =
          b.Get(p + "ffn_up.weight", H, Config::kDenseFf, 1, {GgmlType::kQ8_0});
      L.ffn_down = b.Get(p + "ffn_down.weight", Config::kDenseFf, H, 1,
                         {GgmlType::kQ8_0});
      continue;
    }
    L.router = b.Get(p + "ffn_gate_inp.weight", H, Config::kExperts, 1,
                     {GgmlType::kF32});
    L.router_bias =
        b.Get(p + "exp_probs_b.bias", Config::kExperts, 1, 1, {GgmlType::kF32});
    L.gate_exps = b.Get(p + "ffn_gate_exps.weight", H, Config::kExpertFf,
                        Config::kExperts, kExpertTypes);
    L.up_exps = b.Get(p + "ffn_up_exps.weight", H, Config::kExpertFf,
                      Config::kExperts, kExpertTypes);
    L.down_exps = b.Get(p + "ffn_down_exps.weight", Config::kExpertFf, H,
                        Config::kExperts, kExpertTypes);
  }
  if (!b.ok)
    return std::nullopt;
  return w;
}

}  // namespace gufo::models::mimo_v26_flash

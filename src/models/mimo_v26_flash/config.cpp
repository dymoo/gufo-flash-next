#include "src/models/mimo_v26_flash/config.hpp"

#include <algorithm>
#include <cmath>
#include <string_view>
#include <variant>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::mimo_v26_flash {
namespace {

bool Fail(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
  return false;
}

std::vector<std::uint64_t> IntArray(const core::GgufReader& gguf,
                                    std::string_view key) {
  const auto* meta = gguf.FindMetadata(key);
  if (meta == nullptr)
    return {};
  if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value))
    return *u;
  std::vector<std::uint64_t> out;
  if (const auto* s = std::get_if<std::vector<std::int64_t>>(&meta->value))
    for (const auto v : *s)
      out.push_back(v < 0 ? ~0ULL : static_cast<std::uint64_t>(v));
  return out;
}

}  // namespace

std::uint32_t Config::FullLayerCount() const noexcept {
  return static_cast<std::uint32_t>(std::ranges::count(swa, false));
}

std::optional<Config> Config::Parse(const core::GgufReader& gguf,
                                    std::string* error) {
  if (gguf.GetMetadataString("general.architecture") !=
      std::optional<std::string_view>{"mimo2"}) {
    Fail(error, "not a mimo2 artifact");
    return std::nullopt;
  }
  const auto u32 = [&](std::string_view key) {
    return gguf.GetMetadataUint32(key).value_or(0);
  };
  const auto f32 = [&](std::string_view key, float fallback) {
    const auto v = gguf.GetMetadataFloat32(key);
    return v && std::isfinite(*v) ? *v : fallback;
  };
  const std::uint32_t blocks = u32("mimo2.block_count");
  const std::uint32_t nextn = u32("mimo2.nextn_predict_layers");
  // The kernels are compiled for exactly this geometry.
  const struct {
    std::string_view key;
    std::uint32_t want;
  } fixed[] = {
      {"mimo2.embedding_length", kHidden},
      {"mimo2.attention.head_count", kHeads},
      {"mimo2.attention.key_length", kHeadDimQk},
      {"mimo2.attention.value_length", kHeadDimV},
      {"mimo2.rope.dimension_count", kRopeDims},
      {"mimo2.attention.sliding_window", kWindow},
      {"mimo2.expert_count", kExperts},
      {"mimo2.expert_used_count", kExpertsUsed},
      {"mimo2.expert_feed_forward_length", kExpertFf},
      {"mimo2.feed_forward_length", kDenseFf},
      {"mimo2.expert_gating_func", 2},  // sigmoid
  };
  for (const auto& f : fixed) {
    if (u32(f.key) != f.want) {
      Fail(error, "unsupported MiMo geometry: " + std::string(f.key) + "=" +
                      std::to_string(u32(f.key)) + ", kernels need " +
                      std::to_string(f.want));
      return std::nullopt;
    }
  }
  if (blocks < nextn || blocks - nextn != kLayers) {
    Fail(error, "MiMo trunk must have 48 layers");
    return std::nullopt;
  }
  const auto kv_heads = IntArray(gguf, "mimo2.attention.head_count_kv");
  const auto pattern = IntArray(gguf, "mimo2.attention.sliding_window_pattern");
  if (kv_heads.size() < kLayers || pattern.size() < kLayers) {
    Fail(error, "MiMo per-layer head_count_kv/sliding_window_pattern missing");
    return std::nullopt;
  }

  Config c;
  c.vocab = static_cast<std::uint32_t>(
      gguf.GetMetadataStringArray("tokenizer.ggml.tokens").size());
  c.native_context = u32("mimo2.context_length");
  c.rms_eps = f32("mimo2.attention.layer_norm_rms_epsilon", 1e-6f);
  c.rope_base_full = f32("mimo2.rope.freq_base", 1e7f);
  c.rope_base_swa = f32("mimo2.rope.freq_base_swa", 1e4f);
  c.value_scale = f32("mimo2.attention.value_scale", 1.0f);
  c.swa.resize(kLayers);
  c.dense_ffn.resize(kLayers);
  for (std::uint32_t l = 0; l < kLayers; ++l) {
    c.swa[l] = pattern[l] != 0;
    if (kv_heads[l] != (c.swa[l] ? kSwaKvHeads : kFullKvHeads)) {
      Fail(error, "unexpected KV head count at layer " + std::to_string(l));
      return std::nullopt;
    }
    c.dense_ffn[l] =
        !gguf.HasTensor("blk." + std::to_string(l) + ".ffn_gate_inp.weight");
  }
  if (c.vocab == 0 || c.value_scale <= 0.0f) {
    Fail(error, "MiMo vocabulary or value scale missing");
    return std::nullopt;
  }
  return c;
}

}  // namespace gufo::models::mimo_v26_flash

#include "src/models/qwen38_flash_next/config.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>

#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {

struct Reader {
  const core::GgufReader& gguf;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  std::uint32_t U32(std::string_view key, bool required = true,
                    std::uint32_t fallback = 0) {
    const auto value = gguf.GetMetadataUint32(key);
    if (value.has_value()) {
      return *value;
    }
    if (required || gguf.FindMetadata(key) != nullptr) {
      Fail("missing or invalid GGUF integer " + std::string(key));
    }
    return fallback;
  }

  float F32(std::string_view key, float fallback) {
    const auto value = gguf.GetMetadataFloat32(key);
    if (!value || !std::isfinite(*value))
      Fail("missing or invalid GGUF float " + std::string(key));
    return value.value_or(fallback);
  }

  /// Integer arrays of any width; GGUF writers pick the narrowest type.
  std::vector<std::uint64_t> U64Array(std::string_view key,
                                      bool required = true) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<std::uint64_t> out;
    if (meta == nullptr) {
      if (required)
        Fail("missing GGUF array " + std::string(key));
      return out;
    }
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      out = *u;
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (auto v : *s) {
        if (v < 0) {
          Fail("negative GGUF array value " + std::string(key));
          return {};
        }
        out.push_back(static_cast<std::uint64_t>(v));
      }
    } else {
      Fail("GGUF array must contain integers: " + std::string(key));
    }
    return out;
  }
};

}  // namespace

/// LFM2 MoE (LFM2.5-24B-A2B geometry): short-convolution and GQA layers on
/// one residual stream; the layer kind comes from the per-layer KV head count
/// (zero on convolution layers).
static std::optional<Config> Lfm2FromGguf(Reader& r, bool require_trunk) {
  Config c;
  auto key = [](std::string_view suffix) {
    return "lfm2moe." + std::string(suffix);
  };
  c.family = Family::kLfm2Moe;
  c.num_layers_all = c.num_layers = r.U32(key("block_count"));
  c.hidden_size = r.U32(key("embedding_length"));
  c.context_length = r.U32(key("context_length"));
  c.rms_eps = r.F32(key("attention.layer_norm_rms_epsilon"), 1e-5F);
  c.hc_count = 1;
  c.num_heads = r.U32(key("attention.head_count"));
  const auto kv = r.U64Array(key("attention.head_count_kv"));
  for (std::size_t i = 0; i < kv.size() && i < 64; ++i) {
    if (kv[i] == 0)
      continue;
    if (c.num_kv_heads != 0 && kv[i] != c.num_kv_heads)
      r.Fail("lfm2moe attention layers differ in KV heads");
    c.num_kv_heads = static_cast<std::uint32_t>(kv[i]);
    c.attention_layers |= std::uint64_t{1} << i;
  }
  c.head_dim = c.num_heads != 0 ? c.hidden_size / c.num_heads : 0;
  c.rotary_dim = c.head_dim;
  c.rope_theta = r.F32(key("rope.freq_base"), 1000000.0F);
  c.conv_cache = r.U32(key("shortconv.l_cache"));
  c.ssm_conv_kernel = c.conv_cache;
  c.num_experts = r.U32(key("expert_count"));
  c.num_experts_used = r.U32(key("expert_used_count"));
  c.expert_ff = r.U32(key("expert_feed_forward_length"));
  c.dense_layers = r.U32(key("leading_dense_block_count"), false, 0);
  c.dense_ff = r.U32(key("feed_forward_length"));
  // The dense FFNs run on the shared-expert path and scratch.
  c.shared_expert_ff = c.dense_ff;
  if (r.U32(key("expert_gating_func"), false, 2) != 2)
    r.Fail("lfm2moe experts must use sigmoid gating");
  if (!r.ok)
    return std::nullopt;
  // The kernels are specialized to this geometry.
  if (kv.size() != c.num_layers || c.num_layers > 64 || c.hidden_size != 2048 ||
      c.num_heads != 32 || c.num_kv_heads != 8 || c.head_dim != 64 ||
      c.conv_cache != 3 || c.num_experts != 64 || c.num_experts_used != 4 ||
      c.expert_ff != 1536 || c.dense_layers >= c.num_layers ||
      c.dense_ff == 0 || c.context_length == 0 ||
      c.context_length > INT32_MAX ||
      (require_trunk && c.attention_layers == 0)) {
    r.Fail("unsupported lfm2moe kernel geometry");
    return std::nullopt;
  }
  return c;
}

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       bool require_trunk,
                                       std::string* error_msg,
                                       std::uint32_t experts_used) {
  Reader r{gguf, error_msg};
  const std::string arch{
      gguf.GetMetadataString("general.architecture").value_or("")};
  const auto family = FamilyOf(arch);
  if (!family) {
    r.Fail("GGUF architecture is not qwen4exp, qwen35moe or lfm2moe");
    return std::nullopt;
  }
  if (*family == Family::kLfm2Moe) {
    return Lfm2FromGguf(r, require_trunk);
  }

  // qwen4exp and qwen35moe (Qwen3.5-35B-A3B geometry, e.g. MiniCPM-V 4.7)
  // are the same hybrid DeltaNet/gated-attention MoE; qwen35moe has one
  // residual stream, dense attention and no PLE.
  Config c;
  c.family = *family;
  const FamilyTraits& t = c.Traits();
  const bool qwen35moe = c.family == Family::kQwen35Moe;
  const std::string p = arch + ".";
  auto key = [&](std::string_view suffix) { return p + std::string(suffix); };

  c.num_layers_all = r.U32(key("block_count"));
  c.nextn_layers = r.U32(key("nextn_predict_layers"), false, 0);
  if (c.nextn_layers >= c.num_layers_all) {
    r.Fail("nextn_predict_layers must be below block_count");
    return std::nullopt;
  }
  c.num_layers = c.num_layers_all - c.nextn_layers;
  c.hidden_size = r.U32(key("embedding_length"));
  c.context_length = r.U32(key("context_length"));
  c.rms_eps = r.F32(key("attention.layer_norm_rms_epsilon"), 1e-6F);

  c.hc_count = t.hyper_connections ? r.U32(key("hyper_connection.count")) : 1;
  c.hc_low_rank =
      t.hyper_connections ? r.U32(key("hyper_connection.low_rank")) : 0;

  c.full_attention_interval = r.U32(key("full_attention_interval"));
  c.num_heads = r.U32(key("attention.head_count"));
  c.num_kv_heads = r.U32(key("attention.head_count_kv"));
  c.head_dim = r.U32(key("attention.key_length"));
  if (r.U32(key("attention.value_length"), false, c.head_dim) != c.head_dim)
    r.Fail("attention value length must match the supported key length");
  c.rotary_dim = r.U32(key("rope.dimension_count"));
  c.rope_theta = r.F32(key("rope.freq_base"), 10000000.0F);
  {
    const auto sections = r.U64Array(key("rope.dimension_sections"));
    if (sections.size() != 4 || sections[3] != 0) {
      r.Fail("rope.dimension_sections must contain three axes and a zero pad");
    } else {
      std::uint64_t sum = 0;
      for (std::size_t i = 0; i < 3; ++i) {
        if (sections[i] == 0 || sections[i] > c.rotary_dim / 2) {
          r.Fail("invalid mRoPE axis dimension");
          break;
        }
        sum += sections[i];
        c.rope_sections[i] = static_cast<std::uint32_t>(sections[i]);
      }
      if (sum != c.rotary_dim / 2)
        r.Fail("mRoPE sections must sum to half the rotary dimension");
    }
  }

  if (t.indexer) {
    c.indexer_heads = r.U32(key("attention.indexer.head_count"));
    c.indexer_head_dim = r.U32(key("attention.indexer.key_length"));
    c.indexer_top_k = r.U32(key("attention.indexer.top_k"));
  }
  if (t.indexer) {
    // One ratio per layer; zero on linear layers. Every attention layer of
    // this model shares one value, which is all the runtime implements.
    const auto ratios = r.U64Array(key("attention.compress_ratios"));
    if (ratios.size() != c.num_layers_all || c.full_attention_interval == 0) {
      r.Fail("attention.compress_ratios must describe every layer");
    } else {
      for (std::size_t i = 0; i < ratios.size(); ++i) {
        if (i >= c.num_layers) {
          // Older GGUF converters append zero for nextn layers even though
          // their exported attention includes QSA indexer weights. The pinned
          // Flash-Next architecture uses the same ratio-four indexer in MTP.
          if (ratios[i] != 0 && ratios[i] != 4)
            r.Fail("unsupported MTP attention compression ratio");
          continue;
        }
        const bool sparse = !c.IsLinearLayer(i);
        if (!sparse) {
          if (ratios[i] != 0)
            r.Fail("linear layers cannot use attention compression");
        } else if (ratios[i] == 0 || ratios[i] > 4096) {
          r.Fail("attention compression must fit the 4096-row index ring");
        } else if (c.compress_ratio == 0) {
          c.compress_ratio = static_cast<std::uint32_t>(ratios[i]);
        } else if (c.compress_ratio != ratios[i]) {
          r.Fail("attention.compress_ratios differ between attention layers");
        }
      }
    }
  }

  c.ssm_conv_kernel = r.U32(key("ssm.conv_kernel"));
  c.ssm_head_dim = r.U32(key("ssm.state_size"));
  c.ssm_num_k_heads = r.U32(key("ssm.group_count"));
  c.ssm_num_v_heads = r.U32(key("ssm.time_step_rank"));
  c.ssm_inner_size = r.U32(key("ssm.inner_size"));

  c.num_experts = r.U32(key("expert_count"));
  c.num_experts_used = r.U32(key("expert_used_count"));
  // `--experts-used K`: route every MoE block (trunk and MTP) to its top K
  // experts instead of the GGUF's expert_used_count (lossy).
  if (experts_used != 0) {
    if (experts_used > c.num_experts_used)
      r.Fail("--experts-used must be 1..expert_used_count (" +
             std::to_string(c.num_experts_used) + ")");
    else
      c.num_experts_used = experts_used;
  }
  c.expert_ff = r.U32(key("expert_feed_forward_length"));
  c.shared_expert_ff = r.U32(key("expert_shared_feed_forward_length"));

  if (!qwen35moe) {
    const auto layers = r.U64Array(key("ple.layers"), false);
    if (layers.size() > 1) {
      r.Fail("only one PLE layer is supported");
    } else if (layers.size() == 1) {
      if (layers[0] >= c.num_layers) {
        r.Fail("PLE layer lies outside the trunk");
        return std::nullopt;
      }
      c.ple_layer = static_cast<std::int32_t>(layers[0]);
      c.ple_ngram_size = r.U32(key("ple.ngram_size"));
      c.ple_heads_per_ngram = r.U32(key("ple.heads_per_ngram"));
      c.ple_conv_kernel = r.U32(key("ple.conv_kernel"));
      c.ple_eos_token = r.U32(key("ple.eos_token_id"));
      c.ple_head_dim = r.U32(key("embedding_length_per_layer_input"));
      if (c.ple_ngram_size < 2 || c.ple_ngram_size > kMaxPleNgram ||
          c.ple_heads_per_ngram == 0 ||
          c.ple_heads_per_ngram > kMaxPleHeads / (c.ple_ngram_size - 1) ||
          c.ple_head_dim == 0 || c.ple_conv_kernel == 0) {
        r.Fail("unsupported PLE n-gram geometry");
      } else {
        c.ple_heads = (c.ple_ngram_size - 1) * c.ple_heads_per_ngram;
        const auto multipliers = r.U64Array(key("ple.layer_multipliers"));
        const auto offsets = r.U64Array(key("ple.head_offsets"));
        const auto vocab = r.U64Array(key("ple.head_vocab_sizes"));
        if (multipliers.size() != c.ple_ngram_size ||
            offsets.size() != c.ple_heads || vocab.size() != c.ple_heads) {
          r.Fail("PLE hash tables must match the head count");
        } else {
          std::copy_n(multipliers.begin(), c.ple_ngram_size,
                      c.ple_multipliers.begin());
          for (std::uint32_t h = 0; h < c.ple_heads; ++h) {
            if (vocab[h] == 0 || vocab[h] > UINT32_MAX ||
                offsets[h] > UINT32_MAX - vocab[h]) {
              r.Fail("PLE head range does not fit a 32-bit row index");
              break;
            }
            c.ple_head_offsets[h] = static_cast<std::uint32_t>(offsets[h]);
            c.ple_head_vocab[h] = static_cast<std::uint32_t>(vocab[h]);
            c.ple_rows = std::max(c.ple_rows, offsets[h] + vocab[h]);
          }
        }
      }
    }
  }

  if (!r.ok) {
    return std::nullopt;
  }

  // Structural limits of this runtime, not of the format.
  if (c.context_length == 0 || c.context_length > INT32_MAX ||
      !std::isfinite(c.rms_eps) || c.rms_eps <= 0 ||
      !std::isfinite(c.rope_theta) || c.rope_theta <= 0 || c.hidden_size == 0 ||
      (t.hyper_connections && (c.hc_count < 2 || c.hc_low_rank == 0)) ||
      c.full_attention_interval == 0 || c.num_heads == 0 ||
      c.num_kv_heads == 0 || c.num_heads % c.num_kv_heads != 0 ||
      c.head_dim == 0 || c.rotary_dim == 0 || c.rotary_dim % 2 != 0 ||
      c.rotary_dim > c.head_dim ||
      (t.indexer &&
       (c.indexer_heads == 0 || c.indexer_head_dim == 0 ||
        c.rotary_dim > c.indexer_head_dim || c.compress_ratio == 0 ||
        c.indexer_top_k == 0 || c.indexer_top_k % c.compress_ratio != 0)) ||
      c.ssm_conv_kernel == 0 || c.ssm_head_dim == 0 || c.ssm_num_k_heads == 0 ||
      c.ssm_num_v_heads == 0 || c.ssm_num_v_heads % c.ssm_num_k_heads != 0 ||
      c.ssm_inner_size != c.SsmValueDim() || c.num_experts == 0 ||
      c.num_experts_used == 0 || c.num_experts_used > c.num_experts ||
      c.expert_ff == 0 || c.shared_expert_ff == 0) {
    r.Fail("qwen4exp metadata describes an unsupported shape");
    return std::nullopt;
  }
  if (qwen35moe) {
    if (c.num_layers != 40 || c.nextn_layers != 0 || c.hidden_size != 2048 ||
        c.full_attention_interval != 4 || c.num_heads != 16 ||
        c.num_kv_heads != 2 || c.head_dim != 256 || c.rotary_dim != 64 ||
        c.rope_sections != std::array<std::uint32_t, 4>{11, 11, 10, 0} ||
        c.ssm_conv_kernel != 4 || c.ssm_head_dim != 128 ||
        c.ssm_num_k_heads != 16 || c.ssm_num_v_heads != 32 ||
        c.num_experts != 256 || c.num_experts_used != 8 || c.expert_ff != 512 ||
        c.shared_expert_ff != 512) {
      r.Fail("unsupported qwen35moe kernel geometry");
      return std::nullopt;
    }
    if (require_trunk && c.num_layers % c.full_attention_interval != 0) {
      r.Fail("qwen35moe trunk layer count is not a multiple of the interval");
      return std::nullopt;
    }
    return c;
  }
  // These dimensions select specialized attention, indexer, DeltaNet and
  // expert kernels. Reject other profiles before allocating or launching;
  // satisfying divisibility alone does not make a shape executable.
  if (c.num_layers != 48 || c.nextn_layers > 1 || c.hidden_size != 2560 ||
      c.hc_count != 4 || c.hc_low_rank != 320 ||
      c.full_attention_interval != 4 || c.num_heads != 24 ||
      c.num_kv_heads != 2 || c.head_dim != 256 || c.rotary_dim != 64 ||
      c.rope_sections != std::array<std::uint32_t, 4>{11, 11, 10, 0} ||
      c.indexer_heads != 4 || c.indexer_head_dim != 128 ||
      c.indexer_top_k != 2048 || c.compress_ratio != 4 ||
      c.ssm_conv_kernel != 4 || c.ssm_head_dim != 128 ||
      c.ssm_num_k_heads != 16 || c.ssm_num_v_heads != 48 ||
      c.num_experts != 512 || c.num_experts_used > 10 || c.expert_ff != 640 ||
      // 1280: shared experts widened by top-k healing (expert-prune lane).
      (c.shared_expert_ff != 640 && c.shared_expert_ff != 1280) ||
      (c.ple_layer >= 0 &&
       (c.ple_ngram_size != 3 || c.ple_heads_per_ngram != 8 ||
        c.ple_head_dim != 160 || c.ple_conv_kernel != 4))) {
    r.Fail("unsupported Qwen3.8-Flash-Next kernel geometry");
    return std::nullopt;
  }
  if (require_trunk) {
    if (c.num_layers == 0 || c.num_layers % c.full_attention_interval != 0) {
      r.Fail("qwen4exp trunk layer count is not a multiple of the interval");
      return std::nullopt;
    }
    if (c.ple_layer >= 0 &&
        !c.IsLinearLayer(static_cast<std::uint32_t>(c.ple_layer))) {
      r.Fail("PLE layer must be a linear attention layer");
      return std::nullopt;
    }
  }
  return c;
}

std::optional<MoeRouting> MoeRouting::Parse(std::string_view spec,
                                            std::uint32_t used,
                                            std::string* error_msg) {
  MoeRouting r;
  if (spec == "exact")
    return r;
  const auto fail = [&]() -> std::optional<MoeRouting> {
    if (error_msg != nullptr)
      *error_msg = "invalid --moe-routing '" + std::string(spec) +
                   "': use exact, affinity:KEEP:MARGIN, mass:KEEP:TARGET or "
                   "budget:KEEP:EXPERTS with KEEP 1.." +
                   std::to_string(used);
    return std::nullopt;
  };
  const auto a = spec.find(':');
  const auto b = a == spec.npos ? a : spec.find(':', a + 1);
  if (b == spec.npos)
    return fail();
  const auto name = spec.substr(0, a);
  const auto keep = spec.substr(a + 1, b - a - 1);
  const auto value = spec.substr(b + 1);
  const auto whole = [](auto parsed, std::string_view text) {
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
  };
  if (!whole(std::from_chars(keep.data(), keep.data() + keep.size(), r.keep),
             keep) ||
      r.keep == 0 || r.keep > used)
    return fail();
  if (name == "budget") {
    r.mode = Mode::kBudget;
    if (!whole(std::from_chars(value.data(), value.data() + value.size(),
                               r.budget),
               value) ||
        r.budget == 0)
      return fail();
    return r;
  }
  if (!whole(
          std::from_chars(value.data(), value.data() + value.size(), r.value),
          value) ||
      !std::isfinite(r.value))
    return fail();
  if (name == "affinity" && r.value >= 0.0F)
    r.mode = Mode::kAffinity;
  else if (name == "mass" && r.value > 0.0F && r.value <= 1.0F)
    r.mode = Mode::kMass;
  else
    return fail();
  return r;
}

std::optional<SteeringAdapter> SteeringAdapter::Load(
    std::string name, const std::string& json_path, std::string* error_msg) {
  const auto fail = [&](const std::string& message) {
    if (error_msg != nullptr)
      *error_msg = "adapter '" + json_path + "': " + message;
    return std::nullopt;
  };
  static_assert(std::endian::native == std::endian::little);
  std::ifstream meta(json_path);
  if (!meta)
    return fail("cannot read metadata");
  std::ostringstream text;
  text << meta.rdbuf();
  json::Value v;
  try {
    v = json::parse(text.str());
  } catch (const std::exception& e) {
    return fail(e.what());
  }
  const std::string format = v.is_object() ? v.member_str("format") : "";
  const bool v2 = format == "gufo steering adapter v2";
  if ((!v2 && format != "gufo steering adapter v1") ||
      v.member_str("kind") != "directional ablation")
    return fail(
        "not a 'gufo steering adapter v1' or 'v2' directional ablation");
  const std::size_t hidden = v.member_size("hidden_size", 0);
  const double alpha = v.member_double("alpha", 0.0);
  const std::string file = v.member_str("direction_file");
  if (hidden == 0 || hidden > 65536 || !std::isfinite(alpha) || alpha <= 0.0 ||
      alpha > 4.0 || file.empty())
    return fail("needs hidden_size, alpha in (0, 4] and direction_file");
  SteeringAdapter a{std::move(name),
                    static_cast<float>(alpha),
                    std::vector<float>(hidden),
                    {}};
  if (const auto* layers = v.find("routed_expert_layers"); layers != nullptr) {
    if (!layers->is_array())
      return fail("routed_expert_layers must be an array of layer indices");
    a.routed_layers = 0;
    for (const auto& layer : layers->items()) {
      const double i = layer.is_number() ? layer.as_double() : -1.0;
      if (i < 0.0 || i >= 64.0 || std::floor(i) != i)
        return fail("routed_expert_layers must hold layers 0..63");
      a.routed_layers |= std::uint64_t{1} << static_cast<unsigned>(i);
    }
  }
  if (v2) {
    // v2 names every residual-stream writer it projects; v1 implies the
    // main-layer block outputs only.
    const auto* writers = v.find("writers");
    if (writers == nullptr || !writers->is_array())
      return fail("v2 needs a writers array");
    a.writers = 0;
    for (const auto& w : writers->items()) {
      const std::string s = w.is_string() ? w.str() : "";
      const std::uint8_t bit = s == "blocks"      ? steer_writer::kBlocks
                               : s == "embedding" ? steer_writer::kEmbedding
                               : s == "ple"       ? steer_writer::kPle
                               : s == "mtp"       ? steer_writer::kMtp
                                                  : 0;
      if (bit == 0)
        return fail("writers holds blocks, embedding, ple or mtp, not '" + s +
                    "'");
      a.writers |= bit;
    }
    if ((a.writers & steer_writer::kBlocks) == 0)
      return fail("writers must include blocks");
  }
  const auto path = std::filesystem::path(json_path).parent_path() / file;
  // Stat before reading: a FIFO, a device such as /dev/zero or a huge file
  // must not block or exhaust memory while the adapter loads.
  std::error_code status;
  std::vector<char> raw(hidden * sizeof(float));
  if (!std::filesystem::is_regular_file(path, status) ||
      std::filesystem::file_size(path, status) != raw.size() || status)
    return fail(path.string() + " must be a regular file of " +
                std::to_string(hidden) + " float32 values");
  std::ifstream in(path, std::ios::binary);
  if (!in.read(raw.data(), static_cast<std::streamsize>(raw.size())))
    return fail("cannot read " + path.string());
  std::memcpy(a.direction.data(), raw.data(), raw.size());
  double norm = 0.0;
  for (const float x : a.direction)
    norm += static_cast<double>(x) * x;
  if (!std::isfinite(norm) || std::abs(std::sqrt(norm) - 1.0) > 1e-3)
    return fail("direction is not a unit vector");
  crypto::Sha256Hasher hash;
  hash.Update({reinterpret_cast<const std::uint8_t*>(&a.alpha), sizeof(float)});
  hash.Update({reinterpret_cast<const std::uint8_t*>(&a.routed_layers),
               sizeof(a.routed_layers)});
  hash.Update({reinterpret_cast<const std::uint8_t*>(raw.data()), raw.size()});
  // v1 digests (block outputs only) stay as they were.
  if (a.writers != steer_writer::kBlocks)
    hash.Update({&a.writers, sizeof(a.writers)});
  a.digest = hash.FinishHex();
  return a;
}

}  // namespace gufo::models::qwen38_flash_next

#include "src/models/qwen38_flash_next/weights.hpp"

#include <initializer_list>
#include <string>
#include <utility>

namespace gufo::models::qwen38_flash_next {
namespace {

using core::GgmlType;

struct Format {
  std::uint32_t block;
  std::uint32_t bytes;
};

/// Storage geometry of the formats this runtime decodes. Unknown types get a
/// zero block, which every validation below rejects.
[[nodiscard]] Format FormatOf(GgmlType type) noexcept {
  switch (type) {
    case GgmlType::kF32:
      return {1, 4};
    case GgmlType::kF16:
    case GgmlType::kBF16:
      return {1, 2};
    case GgmlType::kQ8_0:
      return {32, 34};
    case GgmlType::kQ4_0:
      return {32, 18};
    case GgmlType::kQ5_1:
      return {32, 24};
    case GgmlType::kIQ4_NL:
      return {32, 18};
    case GgmlType::kQ4_K:
      return {256, 144};
    case GgmlType::kQ5_K:
      return {256, 176};
    case GgmlType::kQ6_K:
      return {256, 210};
    case GgmlType::kIQ4_XS:
      return {256, 136};
    case GgmlType::kIQ3_S:
      return {256, 110};
    case GgmlType::kIQ3_XXS:
      return {256, 98};
    default:
      return {0, 0};
  }
}

struct Binder {
  const core::GgufReader& reader;
  std::string* error;
  bool ok{true};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  /// Binds `name` with the exact shape and one of the accepted formats.
  TensorRef Get(const std::string& name, std::uint64_t cols, std::uint64_t rows,
                std::uint64_t experts, std::initializer_list<GgmlType> types,
                bool required = true) {
    TensorRef t;
    const auto* info = reader.FindTensor(name);
    if (info == nullptr) {
      if (required) {
        Fail("missing tensor " + name);
      }
      return t;
    }
    const auto& d = info->dimensions;
    const std::uint64_t got_cols = d.size() > 0 ? d[0] : 1;
    const std::uint64_t got_rows = d.size() > 1 ? d[1] : 1;
    const std::uint64_t got_experts = d.size() > 2 ? d[2] : 1;
    if (got_cols != cols || got_rows != rows || got_experts != experts ||
        d.size() > 3) {
      Fail("tensor " + name + " has shape [" + std::to_string(got_cols) + ", " +
           std::to_string(got_rows) + ", " + std::to_string(got_experts) +
           "], expected [" + std::to_string(cols) + ", " +
           std::to_string(rows) + ", " + std::to_string(experts) + "]");
      return t;
    }
    bool type_ok = false;
    for (auto type : types) {
      type_ok = type_ok || info->type == type;
    }
    const Format format = FormatOf(info->type);
    if (!type_ok || format.block == 0 || cols % format.block != 0) {
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
    // Locate the shard so disk readers can address the payload directly, and
    // so a payload running past its shard is caught here rather than as a
    // fault later.
    const auto regions = reader.GetMappedRegions();
    const auto address = reinterpret_cast<std::uintptr_t>(info->data);
    bool inside = false;
    for (std::uint32_t i = 0; i < regions.size(); ++i) {
      const auto base = reinterpret_cast<std::uintptr_t>(regions[i].data);
      if (address >= base && address < base + regions[i].size) {
        t.shard = i;
        t.file_offset = address - base;
        inside = t.SizeBytes() <= regions[i].size - t.file_offset;
        break;
      }
    }
    if (!inside) {
      Fail("tensor " + name + " is truncated");
      return TensorRef{};
    }
    return t;
  }

  /// Routed experts stored in layer `p`: the router's rows when they lie in
  /// [num_experts_used, num_experts], else num_experts (Get then reports the
  /// mismatched shape).
  std::uint64_t LayerExperts(const std::string& p, const Config& c) const {
    const auto* info = reader.FindTensor(p + "ffn_gate_inp.weight");
    if (info == nullptr || info->dimensions.size() < 2)
      return c.num_experts;
    const std::uint64_t rows = info->dimensions[1];
    return rows >= c.num_experts_used && rows <= c.num_experts ? rows
                                                               : c.num_experts;
  }

  /// qwen35moe has one residual stream: its "mixer" is only the pre-norm.
  HcMixer Norm(const std::string& name, const Config& c) {
    HcMixer m;
    m.norm = Get(name, c.hidden_size, 1, 1, {GgmlType::kF32});
    return m;
  }

  HcMixer Mixer(const std::string& prefix, const Config& c, bool with_inject) {
    HcMixer m;
    const std::uint64_t hc_dim = c.HcDim();
    m.norm = Get(prefix + "_norm.weight", hc_dim, 1, 1, {GgmlType::kF32});
    m.down =
        Get(prefix + "_down.weight", hc_dim, c.hc_low_rank, 1,
            {GgmlType::kQ8_0, GgmlType::kBF16, GgmlType::kF16, GgmlType::kF32});
    m.up =
        Get(prefix + "_up.weight", c.hc_low_rank, hc_dim, 1,
            {GgmlType::kQ8_0, GgmlType::kBF16, GgmlType::kF16, GgmlType::kF32});
    if (with_inject) {
      m.inject = Get(prefix + "_inject.weight", hc_dim, c.hc_count, 1,
                     {GgmlType::kF32, GgmlType::kQ8_0, GgmlType::kBF16});
    }
    return m;
  }

  /// lfm2moe: short convolutions map onto the SSM slots (in_proj as
  /// ssm_qkv, the [cache, hidden] taps as ssm_conv1d, out_proj as ssm_out),
  /// the leading dense SwiGLU FFNs onto the shared-expert slots.
  LayerWeights LfmLayer(const Config& c, std::uint32_t il, MixerKind mixer,
                        FfnKind ffn) {
    LayerWeights l;
    l.mixer = mixer;
    l.ffn = ffn;
    const bool linear = l.linear();
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::uint64_t hidden = c.hidden_size;
    // Q4_0 (Liquid's QAD file) becomes Q8_0 dense and Q5_1 down experts at
    // upload, both exactly; its gate/up experts stay Q4_0.
    const auto dense = {GgmlType::kQ8_0, GgmlType::kQ4_0, GgmlType::kBF16,
                        GgmlType::kF16, GgmlType::kF32};
    const auto experts = {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ5_1,
                          GgmlType::kQ8_0, GgmlType::kQ4_0};
    l.hc_attn = Norm(p + "attn_norm.weight", c);
    l.hc_ffn = Norm(p + "ffn_norm.weight", c);
    if (linear) {
      l.ssm_qkv =
          Get(p + "shortconv.in_proj.weight", hidden, 3 * hidden, 1, dense);
      l.ssm_conv1d = Get(p + "shortconv.conv.weight", c.conv_cache, hidden, 1,
                         {GgmlType::kF32});
      l.ssm_out =
          Get(p + "shortconv.out_proj.weight", hidden, hidden, 1, dense);
    } else {
      l.attn_q = Get(p + "attn_q.weight", hidden, c.AttentionQDim(), 1, dense);
      l.attn_k = Get(p + "attn_k.weight", hidden, c.AttentionKvDim(), 1, dense);
      l.attn_v = Get(p + "attn_v.weight", hidden, c.AttentionKvDim(), 1, dense);
      l.attn_out =
          Get(p + "attn_output.weight", c.AttentionQDim(), hidden, 1, dense);
      l.attn_q_norm =
          Get(p + "attn_q_norm.weight", c.head_dim, 1, 1, {GgmlType::kF32});
      l.attn_k_norm =
          Get(p + "attn_k_norm.weight", c.head_dim, 1, 1, {GgmlType::kF32});
    }
    if (ffn == FfnKind::kDense) {
      l.shexp_gate = Get(p + "ffn_gate.weight", hidden, c.dense_ff, 1, dense);
      l.shexp_up = Get(p + "ffn_up.weight", hidden, c.dense_ff, 1, dense);
      l.shexp_down = Get(p + "ffn_down.weight", c.dense_ff, hidden, 1, dense);
      return l;
    }
    l.router = Get(p + "ffn_gate_inp.weight", hidden, c.num_experts, 1,
                   {GgmlType::kF32});
    l.expert_bias =
        Get(p + "exp_probs_b.bias", c.num_experts, 1, 1, {GgmlType::kF32});
    l.ffn_gate_exps = Get(p + "ffn_gate_exps.weight", hidden, c.expert_ff,
                          c.num_experts, experts);
    l.ffn_up_exps = Get(p + "ffn_up_exps.weight", hidden, c.expert_ff,
                        c.num_experts, experts);
    l.ffn_down_exps = Get(p + "ffn_down_exps.weight", c.expert_ff, hidden,
                          c.num_experts, experts);
    return l;
  }

  /// qwen4exp and qwen35moe: hybrid DeltaNet / gated-attention layers with
  /// routed and shared experts.
  LayerWeights Layer(const Config& c, std::uint32_t il, MixerKind mixer,
                     bool ple, bool nextn) {
    LayerWeights l;
    l.mixer = mixer;
    const bool linear = l.linear();
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::uint64_t hidden = c.hidden_size;
    const std::uint64_t hc_dim = c.HcDim();
    // K-quant projections come from llama.cpp's mixes (Q4_K_M and kin).
    const auto dense = {GgmlType::kQ8_0, GgmlType::kQ4_K, GgmlType::kQ5_K,
                        GgmlType::kQ6_K, GgmlType::kBF16, GgmlType::kF16,
                        GgmlType::kF32};
    // The smaller Unsloth quants store routed gate/up as IQ3_S, IQ3_XXS or
    // IQ4_XS and down as IQ4_NL.
    const auto experts = {
        GgmlType::kQ4_K,  GgmlType::kQ5_K,    GgmlType::kQ6_K,
        GgmlType::kQ5_1,  GgmlType::kQ8_0,    GgmlType::kIQ4_XS,
        GgmlType::kIQ3_S, GgmlType::kIQ3_XXS, GgmlType::kIQ4_NL};

    if (c.HyperConnections()) {
      l.hc_attn = Mixer(p + "hc_attn", c, true);
      l.hc_ffn = Mixer(p + "hc_ffn", c, true);
    } else {
      l.hc_attn = Norm(p + "attn_norm.weight", c);
      l.hc_ffn = Norm(p + "post_attention_norm.weight", c);
    }

    if (linear) {
      l.ssm_qkv =
          Get(p + "attn_qkv.weight", hidden, c.SsmConvChannels(), 1, dense);
      l.ssm_gate =
          Get(p + "attn_gate.weight", hidden, c.SsmValueDim(), 1, dense);
      l.ssm_conv1d = Get(p + "ssm_conv1d.weight", c.ssm_conv_kernel,
                         c.SsmConvChannels(), 1, {GgmlType::kF32});
      const auto gates = {GgmlType::kF32,  GgmlType::kBF16, GgmlType::kQ8_0,
                          GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K};
      l.ssm_alpha =
          Get(p + "ssm_alpha.weight", hidden, c.ssm_num_v_heads, 1, gates);
      l.ssm_beta =
          Get(p + "ssm_beta.weight", hidden, c.ssm_num_v_heads, 1, gates);
      l.ssm_dt =
          Get(p + "ssm_dt.bias", c.ssm_num_v_heads, 1, 1, {GgmlType::kF32});
      l.ssm_a = Get(p + "ssm_a", c.ssm_num_v_heads, 1, 1, {GgmlType::kF32});
      l.ssm_norm =
          Get(p + "ssm_norm.weight", c.ssm_head_dim, 1, 1, {GgmlType::kF32});
      l.ssm_out = Get(p + "ssm_out.weight", c.SsmValueDim(), hidden, 1, dense);
    } else {
      l.attn_q =
          Get(p + "attn_q.weight", hidden, 2 * c.AttentionQDim(), 1, dense);
      l.attn_k = Get(p + "attn_k.weight", hidden, c.AttentionKvDim(), 1, dense);
      l.attn_v = Get(p + "attn_v.weight", hidden, c.AttentionKvDim(), 1, dense);
      l.attn_out =
          Get(p + "attn_output.weight", c.AttentionQDim(), hidden, 1, dense);
      l.attn_q_norm =
          Get(p + "attn_q_norm.weight", c.head_dim, 1, 1, {GgmlType::kF32});
      l.attn_k_norm =
          Get(p + "attn_k_norm.weight", c.head_dim, 1, 1, {GgmlType::kF32});
      if (c.HasIndexer()) {
        l.indexer_q = Get(p + "indexer.q_proj.weight", hidden,
                          c.indexer_heads * c.indexer_head_dim, 1, dense);
        l.indexer_k = Get(p + "indexer.k_proj.weight", hidden,
                          c.indexer_head_dim, 1, dense);
        l.indexer_q_norm = Get(p + "indexer.q_norm.weight", c.indexer_head_dim,
                               1, 1, {GgmlType::kF32});
        l.indexer_k_norm = Get(p + "indexer.k_norm.weight", c.indexer_head_dim,
                               1, 1, {GgmlType::kF32});
      }
    }

    if (ple) {
      const std::uint64_t ple_dim = c.PleEmbeddingDim();
      l.ple_key = Get(p + "ple_key.weight", ple_dim, hc_dim, 1, dense);
      l.ple_value = Get(p + "ple_value.weight", ple_dim, hidden, 1, dense);
      l.ple_norm_key =
          Get(p + "ple_norm_key.weight", hc_dim, 1, 1, {GgmlType::kF32});
      l.ple_norm_query =
          Get(p + "ple_norm_query.weight", hc_dim, 1, 1, {GgmlType::kF32});
      l.ple_norm_conv =
          Get(p + "ple_norm_conv.weight", hc_dim, 1, 1, {GgmlType::kF32});
      l.ple_conv1d = Get(p + "ple_conv1d.weight", c.ple_conv_kernel, hc_dim, 1,
                         {GgmlType::kF32});
    }

    // Pruned artifacts keep `expert_count` (the scratch and routing bound)
    // and store fewer experts in a layer; its router rows give the count.
    const std::uint64_t n_experts = LayerExperts(p, c);
    l.router =
        Get(p + "ffn_gate_inp.weight", hidden, n_experts, 1, {GgmlType::kF32});
    l.ffn_gate_exps = Get(p + "ffn_gate_exps.weight", hidden, c.expert_ff,
                          n_experts, experts);
    l.ffn_up_exps =
        Get(p + "ffn_up_exps.weight", hidden, c.expert_ff, n_experts, experts);
    l.ffn_down_exps = Get(p + "ffn_down_exps.weight", c.expert_ff, hidden,
                          n_experts, experts);
    l.shexp_gate_inp =
        Get(p + "ffn_gate_inp_shexp.weight", hidden, 1, 1, {GgmlType::kF32});
    l.shexp_gate =
        Get(p + "ffn_gate_shexp.weight", hidden, c.shared_expert_ff, 1, dense);
    l.shexp_up =
        Get(p + "ffn_up_shexp.weight", hidden, c.shared_expert_ff, 1, dense);
    l.shexp_down =
        Get(p + "ffn_down_shexp.weight", c.shared_expert_ff, hidden, 1, dense);

    if (nextn) {
      l.nextn_enorm =
          Get(p + "nextn.enorm.weight", hidden, 1, 1, {GgmlType::kF32});
      l.nextn_hnorm =
          Get(p + "nextn.hnorm.weight", hc_dim, 1, 1, {GgmlType::kF32});
      l.nextn_eh_proj =
          Get(p + "nextn.eh_proj.weight", 2 * hidden, hidden, 1, dense);
      l.nextn_head = Mixer(p + "nextn.hc_head", c, false);
    }
    return l;
  }
};

}  // namespace

std::size_t TensorRef::RowBytes() const noexcept {
  const Format f = FormatOf(type);
  if (f.block == 0 || cols % f.block != 0) {
    return 0;
  }
  return static_cast<std::size_t>(cols / f.block) * f.bytes;
}

std::optional<ModelWeights> ModelWeights::Bind(const core::GgufReader& reader,
                                               std::string* error_msg,
                                               std::uint32_t first_layer,
                                               std::uint32_t layer_count,
                                               std::uint32_t experts_used) {
  auto config = Config::FromGguf(reader, true, error_msg, experts_used);
  if (!config.has_value()) {
    return std::nullopt;
  }
  if (layer_count != 0) {
    if (first_layer + layer_count > config->num_layers) {
      if (error_msg != nullptr)
        *error_msg = "layer slice exceeds the trunk";
      return std::nullopt;
    }
    const auto ple = config->ple_layer;
    config->ple_layer =
        ple >= static_cast<std::int32_t>(first_layer) &&
                ple < static_cast<std::int32_t>(first_layer + layer_count)
            ? ple - static_cast<std::int32_t>(first_layer)
            : -1;
    config->first_layer = first_layer;
    config->num_layers = layer_count;
  }
  Binder b{reader, error_msg};
  ModelWeights w;
  w.config = *config;
  const Config& c = w.config;
  const auto dense = {GgmlType::kQ8_0, GgmlType::kQ4_K, GgmlType::kQ5_K,
                      GgmlType::kQ6_K, GgmlType::kBF16, GgmlType::kF16,
                      GgmlType::kF32};

  {
    // The vocabulary is the embedding row count; no metadata key carries it.
    const auto* info = reader.FindTensor("token_embd.weight");
    if (info == nullptr || info->dimensions.size() != 2) {
      b.Fail("missing token_embd.weight");
      return std::nullopt;
    }
    w.config.vocab_size = static_cast<std::uint32_t>(info->dimensions[1]);
  }
  w.token_embd =
      b.Get("token_embd.weight", c.hidden_size, c.vocab_size, 1, dense);
  w.output =
      b.Get("output.weight", c.hidden_size, c.vocab_size, 1, dense, false);
  if (w.output.empty()) {
    w.output = w.token_embd;
  }
  w.hc_head = c.HyperConnections() ? b.Mixer("output_hc", c, false)
              : c.family == Family::kLfm2Moe
                  ? b.Norm("token_embd_norm.weight", c)
                  : b.Norm("output_norm.weight", c);
  if (c.ple_layer >= 0) {
    // The converter may pad the table beyond the hashed range, so take the
    // row count from the file and only check that it covers every head.
    const auto* info = reader.FindTensor("per_layer_token_embd.weight");
    const std::uint64_t rows = info != nullptr && info->dimensions.size() == 2
                                   ? info->dimensions[1]
                                   : 0;
    if (rows < c.ple_rows) {
      b.Fail("per_layer_token_embd.weight is missing or too short");
    } else {
      w.ple_table = b.Get("per_layer_token_embd.weight", c.ple_head_dim, rows,
                          1, {GgmlType::kIQ4_NL, GgmlType::kBF16});
    }
  }
  w.layers.reserve(c.num_layers);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    w.layers.push_back(
        c.family == Family::kLfm2Moe
            ? b.LfmLayer(c, c.first_layer + il, c.MixerOf(il), c.FfnOf(il))
            : b.Layer(c, c.first_layer + il, c.MixerOf(il), c.IsPleLayer(il),
                      false));
  }
  if (!b.ok) {
    return std::nullopt;
  }
  return w;
}

std::optional<MtpWeights> MtpWeights::Bind(const core::GgufReader& reader,
                                           const Config& trunk,
                                           std::string* error_msg,
                                           std::uint32_t experts_used) {
  auto config = Config::FromGguf(reader, false, error_msg, experts_used);
  if (!config.has_value()) {
    return std::nullopt;
  }
  Binder b{reader, error_msg};
  if (!config->MtpMatches(trunk)) {
    b.Fail(
        "MTP sidecar architecture or inference constants differ from the "
        "trunk");
    return std::nullopt;
  }
  MtpWeights m;
  m.config = *config;
  m.config.vocab_size = trunk.vocab_size;
  // The draft block is a full-attention layer at index num_layers.
  m.block = b.Layer(m.config, m.config.num_layers,
                    m.config.Traits().attention_mixer, false, true);
  if (!b.ok) {
    return std::nullopt;
  }
  return m;
}

std::optional<DsparkWeights> DsparkWeights::Bind(const core::GgufReader& reader,
                                                 const Config& trunk,
                                                 std::string* error_msg) {
  Binder b{reader, error_msg};
  const auto u32 = [&](const char* key) {
    const auto v = reader.GetMetadataUint32(key);
    if (!v)
      b.Fail(std::string("missing DSpark metadata ") + key);
    return v.value_or(0);
  };
  if (reader.GetMetadataString("general.architecture").value_or("") !=
      "dflash") {
    b.Fail("DSpark drafter must be a dflash GGUF");
    return std::nullopt;
  }
  DsparkWeights d;
  const std::uint32_t hidden = u32("dflash.embedding_length");
  const std::uint32_t layers = u32("dflash.block_count");
  d.block_size = u32("dflash.block_size");
  d.heads = u32("dflash.attention.head_count");
  d.kv_heads = u32("dflash.attention.head_count_kv");
  d.head_dim = u32("dflash.attention.key_length");
  d.ffn = u32("dflash.feed_forward_length");
  d.mask_token = static_cast<std::int32_t>(u32("tokenizer.ggml.mask_token_id"));
  d.rope_theta = reader.GetMetadataFloat32("dflash.rope.freq_base").value_or(0);
  d.rms_eps =
      reader.GetMetadataFloat32("dflash.attention.layer_norm_rms_epsilon")
          .value_or(1e-6F);
  if (const auto* meta = reader.FindMetadata("dflash.target_layers")) {
    if (const auto* v = std::get_if<std::vector<std::uint64_t>>(&meta->value))
      d.target_layers.assign(v->begin(), v->end());
    else if (const auto* s =
                 std::get_if<std::vector<std::int64_t>>(&meta->value))
      d.target_layers.assign(s->begin(), s->end());
  }
  // The kernels reuse the target's 256-wide heads and plain NEOX rotary
  // over the whole head; sample_from_anchor: row 0 already predicts.
  if (!b.ok || hidden != trunk.hidden_size || d.head_dim != 256 ||
      d.heads * d.head_dim != hidden || d.kv_heads == 0 ||
      d.heads % d.kv_heads != 0 || d.block_size < 2 || layers == 0 ||
      d.target_layers.empty() || d.rope_theta <= 0 ||
      !reader.GetMetadataBool("dflash.sample_from_anchor").value_or(false) ||
      d.mask_token < 0 ||
      static_cast<std::uint32_t>(d.mask_token) >= trunk.vocab_size) {
    b.Fail("unsupported DSpark drafter geometry");
    return std::nullopt;
  }
  for (const auto l : d.target_layers) {
    if (l == 0 || l >= trunk.num_layers)
      b.Fail("DSpark target layer outside the trunk");
  }
  const auto vocab = trunk.vocab_size;
  const auto dense = {GgmlType::kBF16, GgmlType::kF16, GgmlType::kQ8_0,
                      GgmlType::kF32};
  const auto f32 = {GgmlType::kF32};
  const std::uint64_t kv = std::uint64_t{d.kv_heads} * d.head_dim;
  const std::uint64_t features = std::uint64_t{hidden} * d.target_layers.size();
  d.fc = b.Get("fc.weight", features, hidden, 1, dense);
  d.enc_norm = b.Get("enc.output_norm.weight", hidden, 1, 1, f32);
  d.out_norm = b.Get("output_norm.weight", hidden, 1, 1, f32);
  const auto* w1 = reader.FindTensor("markov_w1.weight");
  d.rank = w1 != nullptr && !w1->dimensions.empty()
               ? static_cast<std::uint32_t>(w1->dimensions[0])
               : 0;
  // The Markov step kernel reads BF16 rows.
  d.markov_w1 = b.Get("markov_w1.weight", d.rank, vocab, 1, {GgmlType::kBF16});
  d.markov_w2 = b.Get("markov_w2.weight", d.rank, vocab, 1, {GgmlType::kBF16});
  d.conf_w = b.Get("conf_proj.weight", hidden + d.rank, 1, 1, dense);
  d.conf_b = b.Get("conf_proj.bias", 1, 1, 1, f32);
  for (std::uint32_t i = 0; i < layers; ++i) {
    const std::string p = "blk." + std::to_string(i) + ".";
    d.blocks.push_back({
        .attn_norm = b.Get(p + "attn_norm.weight", hidden, 1, 1, f32),
        .q = b.Get(p + "attn_q.weight", hidden, hidden, 1, dense),
        .k = b.Get(p + "attn_k.weight", hidden, kv, 1, dense),
        .v = b.Get(p + "attn_v.weight", hidden, kv, 1, dense),
        .q_norm = b.Get(p + "attn_q_norm.weight", d.head_dim, 1, 1, f32),
        .k_norm = b.Get(p + "attn_k_norm.weight", d.head_dim, 1, 1, f32),
        .out = b.Get(p + "attn_output.weight", hidden, hidden, 1, dense),
        .ffn_norm = b.Get(p + "ffn_norm.weight", hidden, 1, 1, f32),
        .gate = b.Get(p + "ffn_gate.weight", hidden, d.ffn, 1, dense),
        .up = b.Get(p + "ffn_up.weight", hidden, d.ffn, 1, dense),
        .down = b.Get(p + "ffn_down.weight", d.ffn, hidden, 1, dense),
    });
  }
  if (!b.ok || d.rank == 0 || d.rank % 32 != 0 || d.rank > 512 ||
      d.blocks.size() > 8)
    return std::nullopt;
  return d;
}

}  // namespace gufo::models::qwen38_flash_next

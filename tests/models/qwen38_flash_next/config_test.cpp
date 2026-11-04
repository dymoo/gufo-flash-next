#include "src/models/qwen38_flash_next/config.hpp"

#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace qfn = gufo::models::qwen38_flash_next;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

void CheckSidecarCompatibility() {
  qfn::Config trunk;
  trunk.num_layers = 48;
  trunk.context_length = 262144;
  trunk.hidden_size = 2560;
  trunk.hc_count = 4;
  trunk.hc_low_rank = 320;
  trunk.num_experts = 512;
  trunk.num_experts_used = 10;
  trunk.expert_ff = trunk.shared_expert_ff = 640;
  trunk.num_heads = 24;
  trunk.num_kv_heads = 2;
  trunk.head_dim = 256;
  trunk.rotary_dim = 64;
  trunk.rope_theta = 1e7F;
  trunk.rope_sections = {11, 11, 10, 0};
  auto sidecar = trunk;
  sidecar.nextn_layers = 1;
  Require(sidecar.MtpMatches(trunk), "matching MTP constants rejected");
  for (auto member : {&qfn::Config::hc_low_rank, &qfn::Config::context_length,
                      &qfn::Config::rotary_dim, &qfn::Config::num_experts_used,
                      &qfn::Config::shared_expert_ff}) {
    auto wrong = sidecar;
    ++(wrong.*member);
    Require(!wrong.MtpMatches(trunk),
            "dimension-compatible MTP with different semantics accepted");
  }
  for (auto member : {&qfn::Config::rms_eps, &qfn::Config::rope_theta}) {
    auto wrong = sidecar;
    wrong.*member *= 2.0F;
    Require(!wrong.MtpMatches(trunk),
            "MTP numerical constant mismatch accepted");
  }
  auto wrong = sidecar;
  std::swap(wrong.rope_sections[0], wrong.rope_sections[2]);
  Require(!wrong.MtpMatches(trunk), "MTP RoPE partition mismatch accepted");
  wrong = sidecar;
  wrong.family = qfn::Family::kQwen35Moe;
  Require(!wrong.MtpMatches(trunk), "MTP from another family accepted");
  sidecar.ple_layer = -1;
  trunk.ple_layer = 1;
  Require(sidecar.MtpMatches(trunk),
          "trunk-only PLE incorrectly required in MTP");
}

using Value =
    std::variant<std::uint32_t, float, std::string, std::vector<std::uint64_t>,
                 std::vector<std::int64_t>, std::vector<double>>;
using Metadata = std::map<std::string, Value>;

// Minimal metadata-only GGUF: no model or GPU is needed to test the loader.
std::optional<qfn::Config> Parse(const Metadata& fields, bool trunk = true) {
  std::vector<std::uint8_t> bytes;
  const auto pod = [&]<class T>(T value) {
    const auto begin = bytes.size();
    bytes.resize(begin + sizeof(value));
    std::memcpy(bytes.data() + begin, &value, sizeof(value));
  };
  const auto text = [&](const std::string& value) {
    pod(std::uint64_t{value.size()});
    bytes.insert(bytes.end(), value.begin(), value.end());
  };
  pod(std::uint32_t{0x46554747});
  pod(std::uint32_t{3});
  pod(std::uint64_t{0});
  pod(std::uint64_t{fields.size()});
  for (const auto& [key, value] : fields) {
    text(key);
    std::visit(
        [&](const auto& item) {
          using T = std::decay_t<decltype(item)>;
          if constexpr (std::is_same_v<T, std::uint32_t>) {
            pod(std::uint32_t{4});
            pod(item);
          } else if constexpr (std::is_same_v<T, float>) {
            pod(std::uint32_t{6});
            pod(item);
          } else if constexpr (std::is_same_v<T, std::string>) {
            pod(std::uint32_t{8});
            text(item);
          } else {
            pod(std::uint32_t{9});
            using E = typename T::value_type;
            pod(std::uint32_t{std::is_same_v<E, std::uint64_t>  ? 10U
                              : std::is_same_v<E, std::int64_t> ? 11U
                                                                : 12U});
            pod(std::uint64_t{item.size()});
            for (const auto entry : item)
              pod(entry);
          }
        },
        value);
  }
  bytes.resize((bytes.size() + 31) / 32 * 32);
  std::string error;
  const auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
  assert(reader && error.empty());
  const auto result = qfn::Config::FromGguf(*reader, trunk, &error);
  assert(result.has_value() == error.empty());
  return result;
}

Metadata ValidMetadata() {
  Metadata fields{{"general.architecture", std::string{"qwen4exp"}}};
  for (const auto& [key, value] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
           {"block_count", 48},
           {"embedding_length", 2560},
           {"context_length", 262144},
           {"hyper_connection.count", 4},
           {"hyper_connection.low_rank", 320},
           {"full_attention_interval", 4},
           {"attention.head_count", 24},
           {"attention.head_count_kv", 2},
           {"attention.key_length", 256},
           {"rope.dimension_count", 64},
           {"attention.indexer.head_count", 4},
           {"attention.indexer.key_length", 128},
           {"attention.indexer.top_k", 2048},
           {"ssm.conv_kernel", 4},
           {"ssm.state_size", 128},
           {"ssm.group_count", 16},
           {"ssm.time_step_rank", 48},
           {"ssm.inner_size", 6144},
           {"expert_count", 512},
           {"expert_used_count", 10},
           {"expert_feed_forward_length", 640},
           {"expert_shared_feed_forward_length", 640}}) {
    fields[std::string{"qwen4exp."} + key] = value;
  }
  fields["qwen4exp.attention.layer_norm_rms_epsilon"] = 1e-6F;
  fields["qwen4exp.rope.freq_base"] = 1e7F;
  fields["qwen4exp.rope.dimension_sections"] =
      std::vector<std::uint64_t>{11, 11, 10, 0};
  std::vector<std::uint64_t> ratios(48);
  for (std::size_t i = 3; i < ratios.size(); i += 4)
    ratios[i] = 4;
  fields["qwen4exp.attention.compress_ratios"] = ratios;
  return fields;
}

// Each architecture parses to its family; every consumer's question (layer
// kinds, capabilities) is answered by that family's traits row.
void CheckFamilyTable() {
  for (const auto& row : qfn::kFamilies) {
    Require(&qfn::TraitsOf(row.family) == &row, "family rows out of order");
    Require(qfn::FamilyOf(row.architecture) == row.family,
            "architecture does not map to its family");
    Require(!qfn::IsLinearMixer(row.attention_mixer) &&
                qfn::IsLinearMixer(row.linear_mixer),
            "family mixers swapped");
  }
  Require(!qfn::FamilyOf("qwen35") && !qfn::FamilyOf("deepseek4") &&
              !qfn::FamilyOf(""),
          "foreign architecture mapped to a Flash-Next family");
  auto wrong = ValidMetadata();
  wrong["general.architecture"] = std::string{"qwen35"};
  Require(!Parse(wrong), "foreign architecture parsed");
}

// Facts every family must keep consistent with its traits.
void CheckFamilyInvariants(const qfn::Config& c) {
  Require(c.HasIndexer() == (c.compress_ratio != 0) &&
              c.HasIndexer() == (c.indexer_heads != 0),
          "no-indexer state disagrees with compress_ratio");
  Require(c.HyperConnections() == (c.hc_count > 1),
          "hyper-connections disagree with hc_count");
  for (std::uint32_t l = 0; l < c.num_layers; ++l)
    Require(qfn::IsLinearMixer(c.MixerOf(l)) == c.IsLinearLayer(l),
            "layer mixer disagrees with the layer layout");
}

void CheckMalformedMetadata() {
  const auto valid = ValidMetadata();
  assert(Parse(valid));
  {
    const auto c = Parse(valid);
    const auto& t = c->Traits();
    Require(c->family == qfn::Family::kQwen4Exp && c->HyperConnections() &&
                c->HasIndexer() && c->compress_ratio == 4 && t.shared_expert &&
                t.gdn_gate == qfn::GdnGate::kSigmoid &&
                t.vision == qfn::VisionStack::kQwen,
            "qwen4exp family");
    Require(c->MixerOf(0) == qfn::MixerKind::kGatedDeltaNet &&
                c->MixerOf(3) == qfn::MixerKind::kGatedAttention &&
                c->FfnOf(0) == qfn::FfnKind::kRoutedMoe &&
                c->FfnOf(47) == qfn::FfnKind::kRoutedMoe,
            "qwen4exp layer kinds");
    CheckFamilyInvariants(*c);
  }
  for (const auto key : {"qwen4exp.rope.dimension_sections",
                         "qwen4exp.attention.compress_ratios"}) {
    auto wrong = valid;
    wrong.erase(key);
    assert(!Parse(wrong));
    for (const Value value :
         {Value{std::uint32_t{4}}, Value{std::string{"4"}},
          Value{std::vector<std::uint64_t>{}},
          Value{std::vector<std::int64_t>{-1}}, Value{std::vector<double>{4.0}},
          Value{
              std::vector<double>{std::numeric_limits<double>::quiet_NaN()}}}) {
      wrong = valid;
      wrong[key] = value;
      assert(!Parse(wrong));
    }
  }
  for (const auto sections :
       {std::vector<std::uint64_t>{11, 11, 10},
        std::vector<std::uint64_t>{11, 11, 10, 0, 0},
        std::vector<std::uint64_t>{11, 11, 10, 1},
        std::vector<std::uint64_t>{11, 11, 11, 0},
        std::vector<std::uint64_t>{10, 11, 11, 0},
        std::vector<std::uint64_t>{0, 16, 16, 0},
        std::vector<std::uint64_t>{UINT64_MAX, 11, 10, 0}}) {
    auto wrong = valid;
    wrong["qwen4exp.rope.dimension_sections"] = sections;
    assert(!Parse(wrong));
  }
  for (const auto& [index, value] :
       std::initializer_list<std::pair<std::size_t, std::uint64_t>>{
           {0, 4}, {3, 0}, {7, 8}, {3, 4097}, {3, UINT64_MAX}}) {
    auto wrong = valid;
    auto& ratios = std::get<std::vector<std::uint64_t>>(
        wrong["qwen4exp.attention.compress_ratios"]);
    ratios[index] = value;
    assert(!Parse(wrong));
  }
  for (const auto& [key, value] :
       {std::pair{"attention.indexer.key_length", 32U},
        std::pair{"attention.indexer.head_count", 0U},
        std::pair{"attention.indexer.head_count", 8U},
        std::pair{"attention.indexer.key_length", 256U},
        std::pair{"attention.key_length", 128U},
        std::pair{"attention.value_length", 128U},
        std::pair{"embedding_length", 5120U}, std::pair{"ssm.state_size", 64U},
        std::pair{"expert_count", 2048U}, std::pair{"expert_used_count", 11U},
        std::pair{"attention.indexer.top_k", 2047U},
        std::pair{"nextn_predict_layers", 48U}}) {
    auto wrong = valid;
    wrong[std::string{"qwen4exp."} + key] = value;
    assert(!Parse(wrong));
  }
  for (const Value value :
       {Value{std::string{"4"}}, Value{std::vector<std::int64_t>{-1}},
        Value{std::vector<std::uint64_t>{48}}}) {
    auto wrong = valid;
    wrong["qwen4exp.ple.layers"] = value;
    assert(!Parse(wrong));
  }
  auto sidecar = valid;
  sidecar["qwen4exp.block_count"] = std::uint32_t{49};
  sidecar["qwen4exp.nextn_predict_layers"] = std::uint32_t{1};
  std::get<std::vector<std::uint64_t>>(
      sidecar["qwen4exp.attention.compress_ratios"])
      .push_back(0);
  auto sparse_sidecar = sidecar;
  std::get<std::vector<std::uint64_t>>(
      sparse_sidecar["qwen4exp.attention.compress_ratios"])
      .back() = 4;
  assert(Parse(sparse_sidecar, false));
  std::get<std::vector<std::uint64_t>>(
      sparse_sidecar["qwen4exp.attention.compress_ratios"])
      .back() = 8;
  assert(!Parse(sparse_sidecar, false));
  const auto draft = Parse(sidecar, false);
  const auto trunk = Parse(valid);
  assert(draft && trunk && draft->MtpMatches(*trunk));
}

// qwen35moe (MiniCPM-V 4.7 text): one residual stream, no indexer, silu
// DeltaNet gate, and only its own kernel geometry.
void CheckQwen35Moe() {
  Metadata fields{{"general.architecture", std::string{"qwen35moe"}}};
  for (const auto& [key, value] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
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
           {"expert_shared_feed_forward_length", 512}}) {
    fields[std::string{"qwen35moe."} + key] = value;
  }
  fields["qwen35moe.attention.layer_norm_rms_epsilon"] = 1e-6F;
  fields["qwen35moe.rope.freq_base"] = 1e7F;
  fields["qwen35moe.rope.dimension_sections"] =
      std::vector<std::uint64_t>{11, 11, 10, 0};
  const auto c = Parse(fields);
  Require(c && c->family == qfn::Family::kQwen35Moe && c->hc_count == 1 &&
              !c->HyperConnections() && !c->HasIndexer() &&
              c->compress_ratio == 0 &&
              c->Traits().gdn_gate == qfn::GdnGate::kSilu &&
              c->Traits().shared_expert &&
              c->Traits().vision == qfn::VisionStack::kMtmd &&
              c->num_layers == 40 && c->ple_layer < 0,
          "qwen35moe parse");
  Require(c->IsLinearLayer(0) && !c->IsLinearLayer(3), "qwen35moe layers");
  Require(c->MixerOf(0) == qfn::MixerKind::kGatedDeltaNet &&
              c->MixerOf(3) == qfn::MixerKind::kGatedAttention &&
              c->FfnOf(0) == qfn::FfnKind::kRoutedMoe,
          "qwen35moe layer kinds");
  CheckFamilyInvariants(*c);
  auto wrong = fields;
  wrong["qwen35moe.expert_count"] = std::uint32_t{512};
  Require(!Parse(wrong), "qwen35moe accepted Flash-Next expert geometry");
  Require(Parse(ValidMetadata()) && Parse(ValidMetadata())->Traits().gdn_gate ==
                                        qfn::GdnGate::kSigmoid,
          "Flash-Next must keep the sigmoid DeltaNet gate");
}

void CheckLfm2Moe() {
  Metadata fields{{"general.architecture", std::string{"lfm2moe"}}};
  for (const auto& [key, value] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
           {"block_count", 40},
           {"embedding_length", 2048},
           {"context_length", 131072},
           {"feed_forward_length", 11776},
           {"attention.head_count", 32},
           {"expert_count", 64},
           {"expert_used_count", 4},
           {"expert_feed_forward_length", 1536},
           {"leading_dense_block_count", 2},
           {"expert_gating_func", 2},
           {"shortconv.l_cache", 3}}) {
    fields[std::string{"lfm2moe."} + key] = value;
  }
  // Attention on layers 2, 6, ..., 38 (LFM2.5-24B-A2B).
  std::vector<std::uint64_t> kv(40, 0);
  for (std::size_t l = 2; l < 40; l += 4)
    kv[l] = 8;
  fields["lfm2moe.attention.head_count_kv"] = kv;
  fields["lfm2moe.attention.layer_norm_rms_epsilon"] = 1e-5F;
  fields["lfm2moe.rope.freq_base"] = 1e7F;
  const auto c = Parse(fields);
  Require(c && c->family == qfn::Family::kLfm2Moe && c->hc_count == 1 &&
              !c->HyperConnections() && !c->HasIndexer() &&
              c->compress_ratio == 0 && !c->Traits().shared_expert &&
              c->Traits().vision == qfn::VisionStack::kNone &&
              c->num_kv_heads == 8 && c->head_dim == 64 &&
              c->rotary_dim == 64 && c->dense_layers == 2 &&
              c->shared_expert_ff == 11776 && c->SsmConvChannels() == 2048 &&
              c->SsmValueDim() == 0,
          "lfm2moe parse");
  Require(c->IsLinearLayer(0) && !c->IsLinearLayer(2) && c->IsLinearLayer(3) &&
              !c->IsLinearLayer(38) && c->AttentionLayerCount() == 10,
          "lfm2moe layers");
  Require(c->FfnOf(1) == qfn::FfnKind::kDense &&
              c->FfnOf(2) == qfn::FfnKind::kRoutedMoe,
          "lfm2moe dense layers");
  Require(c->MixerOf(0) == qfn::MixerKind::kShortConv &&
              c->MixerOf(2) == qfn::MixerKind::kAttention,
          "lfm2moe layer kinds");
  CheckFamilyInvariants(*c);
  auto wrong = fields;
  wrong["lfm2moe.expert_gating_func"] = std::uint32_t{1};
  Require(!Parse(wrong), "lfm2moe accepted softmax gating");
  wrong = fields;
  wrong["lfm2moe.expert_feed_forward_length"] = std::uint32_t{1792};
  Require(!Parse(wrong), "lfm2moe accepted an unsupported expert width");
}

void CheckParse() {
  std::string error;
  for (const char* bad :
       {"", "affinity", "affinity:6", "affinity:0:1", "affinity:11:1",
        "affinity:6:-1", "mass:2:0", "mass:2:1.5", "budget:2:0", "budget:2:x",
        "exact:1", "affinity:6:1x", "other:1:1"})
    Require(!qfn::MoeRouting::Parse(bad, 10, &error), bad);
  const auto a = qfn::MoeRouting::Parse("affinity:6:1.5", 10, &error);
  Require(a && a->mode == qfn::MoeRouting::Mode::kAffinity && a->keep == 6 &&
              a->value == 1.5F,
          "affinity parse");
  const auto b = qfn::MoeRouting::Parse("budget:2:128", 10, &error);
  Require(b && b->mode == qfn::MoeRouting::Mode::kBudget && b->keep == 2 &&
              b->budget == 128,
          "budget parse");
  Require(qfn::MoeRouting::Parse("exact", 10, &error)->mode ==
              qfn::MoeRouting::Mode::kExact,
          "exact parse");
}

// `--adapter` files: metadata validation, the direction's size and norm,
// and a digest that changes with either value.
void CheckSteeringAdapter() {
  const auto dir = std::filesystem::temp_directory_path() /
                   ("gufo-adapter-test-" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir);
  const auto write = [&](const std::string& name, const std::string& json,
                         std::vector<float> direction) {
    std::ofstream(dir / (name + ".json")) << json;
    std::ofstream(dir / (name + ".f32"), std::ios::binary)
        .write(reinterpret_cast<const char*>(direction.data()),
               static_cast<std::streamsize>(direction.size() * sizeof(float)));
    return (dir / (name + ".json")).string();
  };
  const auto meta = [](const std::string& name, double alpha,
                       std::size_t hidden) {
    return R"({"format": "gufo steering adapter v1", "kind": )"
           R"("directional ablation", "hidden_size": )" +
           std::to_string(hidden) + R"(, "alpha": )" + std::to_string(alpha) +
           R"(, "direction_file": ")" + name + R"(.f32"})";
  };
  std::vector<float> unit(8, 0.0F);
  unit[3] = 1.0F;
  std::string error;
  const auto a = qfn::SteeringAdapter::Load(
      "abl", write("good", meta("good", 1.9553, 8), unit), &error);
  Require(a && a->name == "abl" && std::abs(a->alpha - 1.9553F) < 1e-6F &&
              a->direction == unit && a->digest.size() == 64,
          "valid adapter rejected");
  const auto b = qfn::SteeringAdapter::Load(
      "abl", write("alpha", meta("alpha", 1.5, 8), unit), &error);
  Require(b && b->digest != a->digest, "adapter digest ignores alpha");
  auto other = unit;
  std::swap(other[3], other[4]);
  const auto c = qfn::SteeringAdapter::Load(
      "abl", write("other", meta("other", 1.9553, 8), other), &error);
  Require(c && c->digest != a->digest, "adapter digest ignores direction");
  Require(a->routed_layers == ~std::uint64_t{0}, "routed layers default");
  auto scoped_meta = meta("good", 1.9553, 8);
  scoped_meta.insert(scoped_meta.size() - 1,
                     R"(, "routed_expert_layers": [2, 47])");
  const auto d = qfn::SteeringAdapter::Load(
      "abl", write("scoped", scoped_meta, unit), &error);
  Require(d &&
              d->routed_layers ==
                  ((std::uint64_t{1} << 2) | (std::uint64_t{1} << 47)) &&
              d->digest != a->digest,
          "routed_expert_layers not parsed into the scope and digest");
  Require(a->writers == qfn::steer_writer::kBlocks, "v1 writers default");
  const auto meta2 = [&](const std::string& name, const std::string& writers) {
    auto m = meta(name, 1.9553, 8);
    m.replace(m.find("v1"), 2, "v2");
    m.insert(m.size() - 1, R"(, "writers": )" + writers);
    return m;
  };
  const auto e = qfn::SteeringAdapter::Load(
      "abl",
      write("v2", meta2("v2", R"(["blocks", "embedding", "ple", "mtp"])"),
            unit),
      &error);
  Require(e &&
              e->writers ==
                  (qfn::steer_writer::kBlocks | qfn::steer_writer::kEmbedding |
                   qfn::steer_writer::kPle | qfn::steer_writer::kMtp) &&
              e->digest != a->digest,
          "v2 writers not parsed into the scope and digest");
  const auto f = qfn::SteeringAdapter::Load(
      "abl", write("v2blocks", meta2("v2blocks", R"(["blocks"])"), unit),
      &error);
  Require(
      f && f->writers == qfn::steer_writer::kBlocks && f->digest == a->digest,
      "a v2 block-output adapter must keep the v1 digest");
  std::vector<float> scaled(8, 0.0F);
  scaled[0] = 2.0F;
  const auto bad = [&](const char* message, const std::string& path) {
    Require(!qfn::SteeringAdapter::Load("abl", path, &error) &&
                error.find(path) != std::string::npos,
            message);
  };
  bad("non-unit direction accepted",
      write("norm", meta("norm", 1.0, 8), scaled));
  bad("truncated direction accepted",
      write("short", meta("short", 1.0, 8), std::vector<float>(7, 0.0F)));
  bad("oversized direction accepted",
      write("long", meta("long", 1.0, 8), std::vector<float>(9, 0.0F)));
  // Anything but a regular file is refused before a read (/dev/zero never
  // ends).
  auto device = meta("device", 1.0, 8);
  device.replace(device.find("device.f32"), 10, "/dev/zero");
  bad("device direction accepted", write("device", device, unit));
  std::filesystem::create_directory(dir / "folder.f32");
  std::ofstream(dir / "folder.json") << meta("folder", 1.0, 8);
  bad("directory direction accepted", (dir / "folder.json").string());
  bad("non-positive alpha accepted", write("zero", meta("zero", 0.0, 8), unit));
  bad("other format accepted",
      write("format", R"({"format": "lora", "hidden_size": 8})", unit));
  auto no_writers = meta("nowriters", 1.0, 8);
  no_writers.replace(no_writers.find("v1"), 2, "v2");
  bad("v2 without writers accepted", write("nowriters", no_writers, unit));
  bad("unknown writer accepted",
      write("lmhead", meta2("lmhead", R"(["blocks", "lm_head"])"), unit));
  bad("writers without blocks accepted",
      write("noblocks", meta2("noblocks", R"(["embedding"])"), unit));
  bad("missing metadata accepted", (dir / "absent.json").string());
  std::filesystem::remove_all(dir);
}

}  // namespace

int main() {
  CheckFamilyTable();
  CheckSidecarCompatibility();
  CheckMalformedMetadata();
  CheckQwen35Moe();
  CheckLfm2Moe();
  CheckParse();
  CheckSteeringAdapter();
  std::cout << "Flash-Next metadata checks passed.\n";
}

#include "src/core/model_admission.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>

#include "src/core/gguf_reader.hpp"

namespace gufo::core {
namespace {

using models::qwen38_flash_next::KvCacheFormat;
using models::qwen38_flash_next::KvCacheFormatName;
using models::qwen38_flash_next::ParseKvCacheFormat;

constexpr std::uint8_t Bit(KvCacheFormat format) {
  return static_cast<std::uint8_t>(1U << static_cast<unsigned>(format));
}
constexpr std::uint8_t kF16 = Bit(KvCacheFormat::kF16);
constexpr std::uint8_t kTurboQuant =
    Bit(KvCacheFormat::kTq4) | Bit(KvCacheFormat::kTq3);
constexpr std::uint8_t kTq8 = Bit(KvCacheFormat::kTq8);
constexpr std::array kKvFormats{KvCacheFormat::kF16, KvCacheFormat::kTq8,
                                KvCacheFormat::kTq4, KvCacheFormat::kTq3};

/// Draft backends by --speculative name; bit i of a draft mask is kDrafts[i].
struct Draft {
  std::string_view name;
  std::string_view model_flag;
};
constexpr std::array<Draft, 3> kDrafts{{{"dflash2", "--dflash-model"},
                                        {"dspark", "--dspark-model"},
                                        {"mtp", "--mtp-model"}}};
constexpr std::uint8_t kDFlash = 1;
constexpr std::uint8_t kDSpark = 2;
constexpr std::uint8_t kMtp = 4;

enum class Vision : std::uint8_t { kNone, kQwen, kMtmd };

/// Capability table row: everything a command may ask of one architecture.
struct Capabilities {
  std::string_view architecture{};  ///< GGUF general.architecture
  std::string_view family{};
  Engine engine{Engine::kQwen};
  TemplateFamily template_family{TemplateFamily::kQwen38};  ///< chat protocol
  bool cli{true};  ///< gufo prompt/chat/bench host it, not only gufo serve
  std::uint8_t kv_cache{kF16};
  /// Lossy KV needs <arch>.attention.key_length equal to this; 0: no rule.
  std::uint32_t lossy_kv_head_width{0};
  std::uint8_t serve_drafts{0};
  std::uint8_t cli_drafts{0};
  /// Drafts that adapt their length and so honour --min-draft-tokens; the
  /// others draft fixed or model-owned blocks and need it at 1.
  std::uint8_t adaptive_drafts{0};
  Vision vision{Vision::kNone};
  /// Image input needs <arch>.embedding_length equal to this; 0: any width.
  std::uint32_t vision_width{0};
  bool moe_routing{false};
  bool adapters{false};
  /// --dense-prefill w4a4 (validated on the qwen4exp projection shapes).
  bool dense_prefill_w4a4{false};
};

// The Flash-Next rows (qwen4exp, qwen35moe, lfm2moe) restate engine facts
// whose source is the model family (src/models/qwen38_flash_next/family.hpp):
// vision stack, tq8 and narrow TurboQuant (plain attention), MTP drafts
// (indexer). Change both together.
constexpr std::array<Capabilities, 6> kTable{{
    {.architecture = "qwen4exp",
     .family = "Qwen3.8-Flash-Next",
     .engine = Engine::kFlashNext,
     .kv_cache = kF16 | kTurboQuant,
     .lossy_kv_head_width = 256,
     .serve_drafts = kMtp,
     .cli_drafts = kMtp,
     .vision = Vision::kQwen,
     .moe_routing = true,
     .adapters = true,
     .dense_prefill_w4a4 = true},
    {.architecture = "qwen35moe",
     .family = "MiniCPM-V 4.7 / Qwen3.6",
     .engine = Engine::kFlashNext,
     .template_family = TemplateFamily::kQwen35,
     .kv_cache = kF16 | kTurboQuant,
     .lossy_kv_head_width = 256,
     .serve_drafts = kDSpark,
     .vision = Vision::kMtmd,
     .moe_routing = true},
    {.architecture = "lfm2moe",
     .family = "LFM2.5",
     .engine = Engine::kFlashNext,
     .template_family = TemplateFamily::kLfm2,
     .kv_cache = kF16 | kTq8 | kTurboQuant,
     .moe_routing = true},
    {.architecture = "deepseek4",
     .family = "DeepSeek V4 Flash",
     .engine = Engine::kDeepSeek,
     .template_family = TemplateFamily::kDeepSeekV4,
     .serve_drafts = kDSpark,
     .cli_drafts = kDSpark},
    {.architecture = "mimo2",
     .family = "MiMo-V2.6-Flash",
     .engine = Engine::kMimo,
     .template_family = TemplateFamily::kMimo,
     .cli = false},
    // Qwen3.8 dense; only the 5120-wide 27B carries a vision tower.
    {.architecture = "qwen35",
     .family = "Qwen3.8",
     .serve_drafts = kDFlash,
     .cli_drafts = kDFlash | kMtp,
     .adaptive_drafts = kMtp,
     .vision = Vision::kQwen,
     .vision_width = 5120},
}};

/// Every other architecture runs on the Qwen engine with the Qwen3.8
/// template, text only.
constexpr Capabilities kOtherArchitecture{.family = "Qwen",
                                          .serve_drafts = kDFlash,
                                          .cli_drafts = kDFlash | kMtp,
                                          .adaptive_drafts = kMtp};

/// "a", "a or b", "a, b or c".
std::string Join(std::span<const std::string_view> names) {
  std::string out;
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (i != 0)
      out += i + 1 == names.size() ? " or " : ", ";
    out += names[i];
  }
  return out;
}

}  // namespace

std::optional<LoadPlan> Admit(const GgufReader& reader,
                              const LoadRequest& request, std::string* error) {
  const auto reject = [error](std::string message) {
    if (error != nullptr)
      *error = std::move(message);
    return std::nullopt;
  };
  const std::string architecture(
      reader.GetMetadataString("general.architecture").value_or(""));
  const auto found = std::ranges::find(kTable, std::string_view(architecture),
                                       &Capabilities::architecture);
  const Capabilities& row = found != kTable.end() ? *found : kOtherArchitecture;
  const bool serve = request.surface == Surface::kServe;
  LoadPlan plan{
      .engine = row.engine,
      .template_family = row.template_family,
      .family = std::string(row.family) + " (" +
                (architecture.empty() ? "no architecture" : architecture) +
                ")"};
  if (!serve && !row.cli)
    return reject(plan.family + " runs only under gufo serve");

  const auto kv = ParseKvCacheFormat(request.kv_cache);
  if (!kv)
    return reject("--kv-cache must be f16, tq8, tq4 or tq3");
  if ((row.kv_cache & Bit(*kv)) == 0) {
    std::vector<std::string_view> formats;
    for (const auto format : kKvFormats)
      if ((row.kv_cache & Bit(format)) != 0)
        formats.push_back(KvCacheFormatName(format));
    return reject("--kv-cache " + std::string(request.kv_cache) + ": " +
                  plan.family + " supports " +
                  (formats.size() == 1 ? "only " : "") + Join(formats));
  }
  if (*kv != KvCacheFormat::kF16 && row.lossy_kv_head_width != 0) {
    const auto width =
        reader.GetMetadataUint32(architecture + ".attention.key_length")
            .value_or(0);
    if (width != row.lossy_kv_head_width)
      return reject("--kv-cache " + std::string(request.kv_cache) + " needs " +
                    std::to_string(row.lossy_kv_head_width) +
                    "-wide attention heads; this " + plan.family +
                    " GGUF has " + std::to_string(width));
  }
  plan.kv_cache = *kv;

  if (!request.speculative.empty() && request.speculative != "off") {
    const auto draft =
        std::ranges::find(kDrafts, request.speculative, &Draft::name);
    if (draft == kDrafts.end())
      return reject("--speculative must be dflash2, dspark, mtp or off");
    const auto bit = static_cast<std::uint8_t>(1U << (draft - kDrafts.begin()));
    const std::uint8_t hosted = serve ? row.serve_drafts : row.cli_drafts;
    const std::string flag = "--speculative " + std::string(draft->name);
    if ((hosted & bit) == 0) {
      std::vector<std::string_view> names{"off"};
      for (std::size_t i = 0; i < kDrafts.size(); ++i)
        if ((hosted & (1U << i)) != 0)
          names.push_back(kDrafts[i].name);
      return reject(flag + ": " +
                    (serve ? "gufo serve" : "gufo prompt/chat/bench") +
                    " runs " + plan.family +
                    (hosted == 0 ? " without speculative decoding"
                                 : " with " + Join(names)));
    }
    if (request.draft_model_path.empty())
      return reject(flag + " requires " + std::string(draft->model_flag));
    if (request.min_draft_tokens != 1 && (row.adaptive_drafts & bit) == 0)
      return reject(flag + " requires --min-draft-tokens 1");
  }

  const std::uint32_t width =
      row.vision_width == 0
          ? 0
          : reader.GetMetadataUint32(architecture + ".embedding_length")
                .value_or(0);
  plan.vision = row.vision != Vision::kNone && width == row.vision_width;
  if (request.image_input || !request.vision_model_path.empty()) {
    const std::string flag =
        request.vision_model_path.empty() ? "--image" : "--mmproj";
    if (row.vision == Vision::kNone)
      return reject(flag + ": " + plan.family + " has no image input");
    if (!plan.vision)
      return reject(flag + ": " + plan.family +
                    " takes image input only at embedding width " +
                    std::to_string(row.vision_width) + "; this GGUF has " +
                    std::to_string(width));
    // gufo prompt/chat prepare only Qwen vision prompts.
    if (!serve && row.vision == Vision::kMtmd)
      return reject(flag + ": gufo prompt/chat cannot feed " + plan.family +
                    " images; use gufo serve");
  }

  if (request.dense_prefill != "f16" && request.dense_prefill != "w4a4")
    return reject("--dense-prefill must be f16 or w4a4");
  if (request.dense_prefill == "w4a4" && !row.dense_prefill_w4a4)
    return reject("--dense-prefill w4a4: " + plan.family +
                  " supports only f16");
  plan.dense_prefill_w4a4 = request.dense_prefill == "w4a4";

  if (request.moe_routing != "exact" && !row.moe_routing)
    return reject("--moe-routing " + std::string(request.moe_routing) + ": " +
                  plan.family + " supports only exact");

  for (const auto& spec : request.adapters) {
    if (!row.adapters)
      return reject("--adapter: " + plan.family + " has no steering adapters");
    const auto equals = spec.find('=');
    std::string name = spec.substr(0, equals);
    if (equals == std::string::npos || name.empty() ||
        equals + 1 == spec.size() ||
        std::ranges::any_of(plan.adapters,
                            [&](const auto& a) { return a.first == name; }))
      return reject("--adapter needs distinct NAME=PATH.json values");
    plan.adapters.emplace_back(std::move(name), spec.substr(equals + 1));
  }

  plan.max_context = request.max_context;
  if (plan.max_context == 0) {
    const auto native =
        reader.GetMetadataUint64(architecture + ".context_length").value_or(0);
    if (native < 2 || native > std::numeric_limits<std::uint32_t>::max())
      return reject(
          "GGUF has no valid native context length; specify --context");
    plan.max_context = static_cast<std::uint32_t>(native);
  }
  return plan;
}

}  // namespace gufo::core

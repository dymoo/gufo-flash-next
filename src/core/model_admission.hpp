#ifndef GUFO_CORE_MODEL_ADMISSION_HPP_
#define GUFO_CORE_MODEL_ADMISSION_HPP_

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/qwen38_flash_next/kv_cache.hpp"

namespace gufo::core {

class GgufReader;

/// Runtime that hosts a model family. `gufo serve`, `gufo prompt`/`chat` and
/// `gufo bench` each have one adapter per engine that instantiates a plan.
enum class Engine : std::uint8_t { kQwen, kFlashNext, kDeepSeek, kMimo };

/// Models that share one chat template and output syntax; the chat protocol
/// (src/cli/serve/chat_protocol.hpp) has one adapter per family.
enum class TemplateFamily : std::uint8_t {
  kQwen38,      ///< Qwen3.8 ChatML (27B, Flash-Next) with an effort instruction
  kQwen35,      ///< Qwen3.5 ChatML (qwen35moe: MiniCPM-V 4.7, Qwen3.6)
  kLfm2,        ///< LFM2.5 (lfm2moe): always thinks, Pythonic tool calls
  kDeepSeekV4,  ///< DeepSeek V4 Flash: DSML tool calls
  kMimo,        ///< MiMo-V2.6-Flash: the model opens its own reasoning
};

/// The command that will instantiate the plan: `gufo serve`, or the
/// in-process `gufo prompt`, `gufo chat` and `gufo bench`.
enum class Surface : std::uint8_t { kServe, kCli };

/// What a command asks of a model, as parsed from its flags. The defaults
/// request nothing beyond plain autoregressive text.
struct LoadRequest {
  Surface surface{Surface::kServe};
  std::uint32_t max_context{0};  ///< 0 selects the GGUF's native context
  std::string_view kv_cache{"f16"};
  std::string_view speculative{};       ///< dflash2, dspark, mtp; "" or off
  std::string_view draft_model_path{};  ///< that backend's --*-model file
  std::uint32_t min_draft_tokens{1};
  std::string_view vision_model_path{};  ///< --mmproj
  bool image_input{false};               ///< --image
  std::string_view moe_routing{"exact"};
  std::span<const std::string> adapters{};  ///< --adapter NAME=PATH values
  std::string_view dense_prefill{"f16"};    ///< --dense-prefill f16|w4a4
};

/// A request admitted against its architecture's capability table: what an
/// engine adapter instantiates.
struct LoadPlan {
  Engine engine{Engine::kQwen};
  /// Every command renders prompts and parses output with this family's
  /// chat protocol.
  TemplateFamily template_family{TemplateFamily::kQwen38};
  std::string family{};  ///< e.g. "LFM2.5 (lfm2moe)", for messages and logs
  std::uint32_t max_context{0};
  models::qwen38_flash_next::KvCacheFormat kv_cache{
      models::qwen38_flash_next::KvCacheFormat::kF16};
  bool vision{false};  ///< this artifact accepts image input
  /// --dense-prefill w4a4: lossy int4 dense projections in prompt processing.
  bool dense_prefill_w4a4{false};
  std::vector<std::pair<std::string, std::string>> adapters{};  ///< NAME, PATH
};

/// Model admission: checks `request` against the capability table row for
/// the GGUF's general.architecture. CPU-only metadata reads, so a rejection
/// happens before any weight upload. Returns nullopt and sets *error to a
/// message naming the flag and the model family.
[[nodiscard]] std::optional<LoadPlan> Admit(const GgufReader& reader,
                                            const LoadRequest& request,
                                            std::string* error);

/// Help for the capability flags, shared by every command that exposes them.
inline constexpr std::string_view kKvCacheHelp =
    "Flash-Next engine attention KV storage: f16, or lossy tq8 (LFM2)/tq4/tq3 "
    "(default: f16)";
inline constexpr std::string_view kDensePrefillHelp =
    "Flash-Next engine dense prompt projections: f16 (exact, default) or "
    "lossy w4a4 (rotated int4 weights and activations)";
inline constexpr std::string_view kMoeRoutingHelp =
    "Flash-Next engine decode routing: exact (default), affinity:KEEP:MARGIN, "
    "mass:KEEP:TARGET or budget:KEEP:EXPERTS; non-exact outputs depend on "
    "co-scheduled requests";

}  // namespace gufo::core

#endif  // GUFO_CORE_MODEL_ADMISSION_HPP_

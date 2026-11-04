#ifndef GUFO_SERVER_INFERENCE_BACKEND_HPP_
#define GUFO_SERVER_INFERENCE_BACKEND_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/cli/serve/text_generation_scheduler.hpp"
#include "src/core/model_admission.hpp"
#include "src/models/qwen/dflash_policy.hpp"

namespace gufo::hip {
class QwenGpuModel;
}

namespace gufo::models::deepseek_v4_flash {
class Model;
}

namespace gufo::models::qwen38_flash_next {
class Model;
}

namespace gufo::models::mimo_v26_flash {
class Model;
}

namespace gufo::tokenization {
class QwenTokenizer;
}

namespace gufo::server {

enum class TextSpeculativeBackend : std::uint8_t {
  kDisabled,
  kDFlash,
  kDSpark,
  kMtp,  ///< Qwen MTP draft block
};

struct TextSpeculativeConfig {
  TextSpeculativeBackend backend{TextSpeculativeBackend::kDisabled};
  std::string draft_model_path;
  std::uint32_t max_draft_tokens{7};
  std::uint32_t min_draft_tokens{1};
  /// Qwen3.8-Flash-Next MTP (`--mtp-policy adaptive`, `--mtp-draft-vocab`).
  bool mtp_adaptive{false};
  std::string mtp_draft_vocab;
  std::uint32_t mtp_draft_vocab_size{32768};
  speculative::DFlashDraftPolicy dflash_policy{
      speculative::DFlashDraftPolicy::kAdaptive};
};

struct TextDiskCacheConfig {
  std::filesystem::path directory;
  std::size_t capacity_bytes{TextRunnerDiskCacheOptions::kDefaultCapacityBytes};
  /// Zero selects the automatic host RAM budget after session allocation.
  std::size_t staging_capacity_bytes{0};
  std::size_t pin_capacity_bytes{
      TextRunnerDiskCacheOptions{}.pin_capacity_bytes};
  std::size_t pin_min_conversations{
      TextRunnerDiskCacheOptions{}.pin_min_conversations};
  std::string model_artifact_fingerprint;
  std::string draft_model_artifact_fingerprint;
};

/// Thread-safe HTTP inference facade over shared immutable GPU model resources
/// and a bounded pool of request-owned executor sessions.
class InferenceBackend final : public TextGenerationBackend {
public:
  using TextGenerationBackend::chat;
  using TextGenerationBackend::complete;
  using TextGenerationBackend::start_chat;

  InferenceBackend();
  ~InferenceBackend() override;

  InferenceBackend(const InferenceBackend&) = delete;
  InferenceBackend& operator=(const InferenceBackend&) = delete;
  InferenceBackend(InferenceBackend&&) = delete;
  InferenceBackend& operator=(InferenceBackend&&) = delete;

  /// Loads weights from a GGUF file. Model admission
  /// (src/core/model_admission.hpp) first checks every flag against the GGUF
  /// architecture's capability table, so an unsupported combination fails
  /// before any weight upload. Returns false and sets *error on failure.
  bool load(const std::string& model_path, std::string* error,
            std::uint32_t max_context = 0, std::size_t session_count = 1,
            TextPrefillPolicy prefill_policy = {},
            TextSchedulerPolicy scheduler_policy = {},
            const TextSpeculativeConfig& speculative_config = {},
            const TextDiskCacheConfig& disk_cache_config = {},
            const std::string& vision_model_path = {},
            const std::string& moe_routing = "exact",
            std::string_view kv_cache = "f16",
            std::span<const std::string> adapters = {},
            std::string_view dense_prefill = "f16",
            std::string_view gdn_state = "f32",
            std::uint32_t routed_prefill_a4 = 0,
            std::uint32_t prefill_capacity = 0, bool routed_down_w4 = false,
            std::uint32_t routed_w4 = 0, std::uint32_t prefill_chunk = 0,
            std::string_view dense_prefill_weights = {},
            std::uint32_t experts_used = 0, std::uint32_t router_norm_k = 0,
            std::string_view prefill_residual = "f32",
            std::string_view gguf_decode_override = {},
            std::uint32_t decode_twin_rows = 4, std::uint32_t fast_experts = 0);

#if defined(ENGINE_ENABLE_HIP)
  /// Installs a previously loaded model without duplicating mapped weights.
  bool load(std::shared_ptr<const hip::QwenGpuModel> model, std::string* error,
            std::uint32_t max_context = 0, std::size_t session_count = 1,
            TextPrefillPolicy prefill_policy = {},
            TextSchedulerPolicy scheduler_policy = {},
            TextSpeculativeConfig speculative_config = {},
            TextDiskCacheConfig disk_cache_config = {});

  /// Installs a previously loaded DeepSeek model with request-owned sessions.
  bool load(std::shared_ptr<models::deepseek_v4_flash::Model> model,
            std::string* error, std::uint32_t max_context = 0,
            std::size_t session_count = 1,
            TextPrefillPolicy prefill_policy = {},
            TextSchedulerPolicy scheduler_policy = {},
            TextSpeculativeConfig speculative_config = {},
            TextDiskCacheConfig disk_cache_config = {});

  /// Installs a previously loaded MiMo-V2.6-Flash model.
  bool load(std::shared_ptr<models::mimo_v26_flash::Model> model,
            std::string* error, std::uint32_t max_context = 0,
            std::size_t session_count = 1,
            TextPrefillPolicy prefill_policy = {},
            TextSchedulerPolicy scheduler_policy = {},
            TextSpeculativeConfig speculative_config = {},
            TextDiskCacheConfig disk_cache_config = {});

  /// Installs a previously loaded Flash-Next engine model with
  /// request-owned sessions, the model's tokenizer, and host-memory
  /// continuation snapshots. `template_family` is the load plan's.
  bool load(
      std::shared_ptr<models::qwen38_flash_next::Model> model,
      std::string* error, std::uint32_t max_context = 0,
      std::size_t session_count = 1, TextPrefillPolicy prefill_policy = {},
      TextSchedulerPolicy scheduler_policy = {},
      TextSpeculativeConfig speculative_config = {},
      TextDiskCacheConfig disk_cache_config = {},
      core::TemplateFamily template_family = core::TemplateFamily::kQwen38);
#endif

  /// Stable model identifier used in API responses.
  [[nodiscard]] std::string model_id() const override;
  [[nodiscard]] std::vector<std::string> model_ids() const override;
  [[nodiscard]] bool ready() const override;
  [[nodiscard]] SamplingDefaults sampling_defaults() const override;
  [[nodiscard]] std::uint32_t max_context() const;
  [[nodiscard]] ReasoningOptions reasoning_defaults() const override;
  [[nodiscard]] const ChatProtocol& chat_protocol() const override;
  [[nodiscard]] MetricsSnapshot metrics_snapshot() const override;
  bool drain() override;
  void set_model_id(const std::string& model_id);
  void set_sampling_defaults(std::size_t max_tokens,
                             const sampling::SamplingConfig& sampling);
  void set_reasoning_defaults(const ReasoningOptions& reasoning);

  /// Plain text completion (no chat framing).
  Result complete(std::string_view prompt, std::size_t max_tokens,
                  const sampling::SamplingConfig& sampling,
                  const CancellationCheck& is_cancelled = {},
                  const TokenCallback& on_token = {},
                  std::string_view client_id = "anonymous",
                  const std::vector<std::string>& stop_sequences = {}) override;

  /// Framed chat conversation; returns the assistant reply.
  Result chat(const ChatRequest& request, std::size_t max_tokens,
              const sampling::SamplingConfig& sampling,
              const CancellationCheck& is_cancelled = {},
              const TokenCallback& on_token = {}) override;

  std::shared_ptr<GenerationRequest> start_chat(
      const ChatRequest& request, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled = {},
      bool stream_output = false) override;

  Result chat(const std::vector<tokenization::ChatMessage>& messages,
              std::size_t max_tokens, const sampling::SamplingConfig& sampling,
              const CancellationCheck& is_cancelled = {});

  /// Token count of raw text (no generation).
  [[nodiscard]] std::size_t count_tokens(std::string_view text) const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_INFERENCE_BACKEND_HPP_

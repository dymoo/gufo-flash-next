#ifndef GUFO_SERVER_TEXT_GENERATION_BACKEND_HPP_
#define GUFO_SERVER_TEXT_GENERATION_BACKEND_HPP_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/reasoning.hpp"
#include "src/core/sampling.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::server {

class ChatProtocol;

/// OpenAI `service_tier`. Flex requests run only on capacity that default
/// requests leave free and are rejected at once when there is none.
enum class ServiceTier : std::uint8_t {
  kDefault,
  kFlex,
};

[[nodiscard]] constexpr const char* ServiceTierName(ServiceTier tier) noexcept {
  return tier == ServiceTier::kFlex ? "flex" : "default";
}

enum class TextGenerationErrorCode : std::uint8_t {
  kQueueFull,
  kClientQueueFull,
  kResourceUnavailable,
  kCapacityExhausted,
  kDraining,
  kDeadlineExceeded,
  kOutputLimit,
  kOutputBackpressure,
  kSchedulerStopping,
  kToolChoiceUnsatisfied,
};

class TextGenerationError final : public std::runtime_error {
public:
  TextGenerationError(TextGenerationErrorCode code, const std::string& message)
      : std::runtime_error(message), code_(code) {}

  [[nodiscard]] TextGenerationErrorCode code() const noexcept { return code_; }
  [[nodiscard]] int http_status() const noexcept {
    switch (code_) {
      case TextGenerationErrorCode::kQueueFull:
      case TextGenerationErrorCode::kClientQueueFull:
      case TextGenerationErrorCode::kResourceUnavailable:
        return 429;
      case TextGenerationErrorCode::kDeadlineExceeded:
        return 408;
      case TextGenerationErrorCode::kOutputLimit:
      case TextGenerationErrorCode::kOutputBackpressure:
      case TextGenerationErrorCode::kSchedulerStopping:
      case TextGenerationErrorCode::kDraining:
      case TextGenerationErrorCode::kCapacityExhausted:
        return 503;
      case TextGenerationErrorCode::kToolChoiceUnsatisfied:
        return 502;
    }
    return 500;
  }
  [[nodiscard]] const char* stable_code() const noexcept {
    switch (code_) {
      case TextGenerationErrorCode::kQueueFull:
        return "queue_full";
      case TextGenerationErrorCode::kClientQueueFull:
        return "client_queue_full";
      case TextGenerationErrorCode::kResourceUnavailable:
        return "resource_unavailable";
      case TextGenerationErrorCode::kCapacityExhausted:
        return "capacity_exhausted";
      case TextGenerationErrorCode::kDraining:
        return "draining";
      case TextGenerationErrorCode::kDeadlineExceeded:
        return "deadline_exceeded";
      case TextGenerationErrorCode::kOutputLimit:
        return "output_limit";
      case TextGenerationErrorCode::kOutputBackpressure:
        return "output_backpressure";
      case TextGenerationErrorCode::kSchedulerStopping:
        return "scheduler_stopping";
      case TextGenerationErrorCode::kToolChoiceUnsatisfied:
        return "tool_choice_unsatisfied";
    }
    return "generation_error";
  }
  [[nodiscard]] bool retryable() const noexcept {
    return code_ != TextGenerationErrorCode::kOutputLimit &&
           code_ != TextGenerationErrorCode::kToolChoiceUnsatisfied;
  }

private:
  TextGenerationErrorCode code_;
};

/// OpenAI `response_format`, enforced by grammar-constrained sampling.
struct ResponseFormat {
  enum class Type : std::uint8_t {
    kText,
    kJsonObject,
    kJsonSchema,
  };
  Type type{Type::kText};
  /// JSON Schema document text (kJsonSchema).
  std::string schema;
};

struct ChatRequest {
  enum class ToolChoice : std::uint8_t {
    kAuto,
    kNone,
    /// At least one call; exactly a call of tool_choice_function when set.
    kRequired,
  };

  ChatRequest() = default;
  explicit ChatRequest(std::vector<tokenization::ChatMessage> chat_messages)
      : messages(std::move(chat_messages)) {}

  std::vector<tokenization::ChatMessage> messages;
  std::vector<tokenization::ChatTool> tools;
  /// Requested public model id, echoed in responses and logs; empty or
  /// model_id() selects the base model, another of model_ids() its steering
  /// adapter, and either with the `-flex` suffix that variant at the flex
  /// tier (see TextGenerationBackend::kFlexSuffix).
  std::string model;
  std::string client_id{"anonymous"};
  ToolChoice tool_choice{ToolChoice::kAuto};
  /// Named `tool_choice` function (with kRequired).
  std::string tool_choice_function;
  /// `parallel_tool_calls`: false stops a constrained tool choice after one
  /// call.
  bool parallel_tool_calls{true};
  ResponseFormat response_format;
  ServiceTier service_tier{ServiceTier::kDefault};
  /// `X-Gufo-No-Queue: 1`: reject instead of queueing when no session is free.
  bool no_queue{false};
  /// `X-Gufo-Queue: 1`: wait for a session under `--admission refuse`.
  bool queue{false};
  ReasoningOptions reasoning;
  bool add_vision_id{false};
  /// Bypass prompt reuse for this request; its completed state may be retained.
  bool cache_prompt{true};
  std::vector<std::string> stop_sequences;
  /// Teacher-forced assistant text, tokenized by the backend into
  /// SamplingConfig::forced_tokens; generation ends after its last token.
  std::string forced_completion;
};

/// One emitted token's raw-model log-probability and top alternatives.
struct TokenLogprob {
  tokenization::TokenId token{0};
  std::string text;
  double logprob{0.0};
  std::vector<TokenLogprob> top;
};

/// Model-agnostic text generation boundary used by the HTTP transport.
///
/// Implementations retain ownership of tokenization, templates, complete
/// continuation state, and execution resources. The transport observes only
/// generated text pieces and aggregate request metrics.
class TextGenerationBackend {
public:
  using CancellationCheck = std::function<bool()>;
  using TokenCallback = std::function<bool(std::string_view)>;

  enum class FinishReason : std::uint8_t {
    kStop,
    kStopSequence,
    kLength,
    kCancelled,
  };

  enum class InitialOutputState : std::uint8_t {
    kAuto,
    kReasoning,
    kContent,
  };

  struct SamplingDefaults {
    /// Zero means generate until EOS or the remaining context is exhausted.
    std::size_t max_tokens{0};
    sampling::SamplingConfig sampling;
  };

  struct Result {
    /// Public model id that served the request (request log).
    std::string model;
    std::string text;
    std::vector<tokenization::TokenId> tokens;
    /// One entry per emitted token when SamplingConfig::top_logprobs >= 0.
    std::vector<TokenLogprob> logprobs;
    std::size_t prompt_tokens{0};
    std::size_t cached_prompt_tokens{0};
    std::size_t cache_restore_bytes{0};
    std::size_t cache_snapshot_bytes{0};
    /// Bytes admitted to background persistence; completion is logged by cache.
    std::size_t cache_disk_queued_bytes{0};
    std::size_t cache_shared_bytes{0};
    /// Shared-prefix snapshots queued while prefilling this request.
    std::size_t cache_shared_prefix_snapshots{0};
    std::size_t cache_shared_prefix_bytes{0};
    std::size_t completion_tokens{0};
    /// Generated reasoning tokens, excluding the closing template delimiter.
    std::size_t reasoning_tokens{0};
    std::size_t draft_tokens{0};
    std::size_t draft_accepted_tokens{0};
    std::size_t prefill_tokens{0};
    std::size_t prefill_chunks{0};
    std::size_t active_decode_prefill_chunks{0};
    std::size_t max_prefill_chunk_tokens{0};
    std::size_t max_consecutive_active_prefill_chunks{0};
    std::size_t configured_active_prefill_tokens{0};
    std::size_t queue_depth_at_submit{0};
    std::size_t client_queue_depth_at_submit{0};
    std::size_t resident_requests_at_admission{0};
    std::size_t requested_logical_concurrency{1};
    std::size_t physical_execution_width{1};
    std::size_t max_buffered_output_bytes{0};
    double queue_ms{0.0};
    double cache_restore_ms{0.0};
    double cache_snapshot_ms{0.0};
    double cache_disk_enqueue_ms{0.0};
    double cache_shared_prefix_ms{0.0};
    double ttft_ms{0.0};
    double mean_inter_token_ms{0.0};
    double max_inter_token_ms{0.0};
    double prefill_ms{0.0};
    double decode_ms{0.0};
    std::string client_id{"anonymous"};
    std::string execution_plan{"serial-c1"};
    std::string prefill_fallback_reason;
    std::string cache_miss_reason;
    std::size_t cache_common_prefix_tokens{0};
    std::size_t cache_checkpoint_tokens{0};
    FinishReason finish_reason{FinishReason::kStop};
    std::string stop_sequence;
    bool incremental_prefill_supported{false};
    bool cache_hit{false};
    bool cache_disk_hit{false};
    bool cancelled{false};
  };

  /// Best-effort scheduler/pool/device gauges for /metrics. Every field
  /// defaults to zero so a backend without a scheduler (a test fake, a
  /// CPU-only build) still exposes a valid, empty snapshot.
  struct MetricsSnapshot {
    std::size_t sessions_configured{0};
    std::size_t sessions_resident{0};
    std::size_t sessions_active{0};
    std::size_t sessions_queued{0};
    std::size_t sessions_queued_flex{0};
    std::size_t sessions_paged_out{0};
    std::size_t sessions_flex{0};
    std::size_t prefilling_default{0};
    /// Sessions flex requests may hold (`--flex-sessions`).
    std::size_t flex_sessions{0};
    std::size_t max_pending{0};
    /// `--admission refuse`.
    bool refuse_when_full{false};
    std::size_t context_tokens{0};
    bool draining{false};
    std::size_t kv_pool_bytes_used{0};
    std::size_t kv_pool_bytes_limit{0};
    std::size_t host_snapshot_bytes{0};
    std::size_t host_snapshot_bytes_limit{0};
    std::size_t disk_cache_bytes{0};
    std::size_t disk_cache_bytes_limit{0};
    std::string kv_cache{"unknown"};
    std::string moe_routing{"unknown"};
    std::string speculative{"off"};
  };

  class GenerationRequest {
  public:
    GenerationRequest() = default;
    virtual ~GenerationRequest() = default;

    GenerationRequest(const GenerationRequest&) = delete;
    GenerationRequest& operator=(const GenerationRequest&) = delete;
    GenerationRequest(GenerationRequest&&) = delete;
    GenerationRequest& operator=(GenerationRequest&&) = delete;

    virtual Result Wait(const TokenCallback& on_token = {}) = 0;
    virtual void Cancel() noexcept = 0;
  };

  TextGenerationBackend() = default;
  virtual ~TextGenerationBackend() = default;

  TextGenerationBackend(const TextGenerationBackend&) = delete;
  TextGenerationBackend& operator=(const TextGenerationBackend&) = delete;
  TextGenerationBackend(TextGenerationBackend&&) = delete;
  TextGenerationBackend& operator=(TextGenerationBackend&&) = delete;

  [[nodiscard]] virtual std::string model_id() const = 0;
  /// Every public id: model_id() first, then its steering adapters
  /// (`--adapter`), which chat requests select by their `model` field.
  [[nodiscard]] virtual std::vector<std::string> model_ids() const {
    return {model_id()};
  }
  /// `<id>-flex` is the variant `<id>` (the base model or one of its steering
  /// adapters) at `service_tier: "flex"`, for callers that choose a model but
  /// cannot set body fields (gateway routes). The tier is scheduling only:
  /// the alias runs the variant's own weights and arithmetic.
  static constexpr std::string_view kFlexSuffix = "-flex";
  /// `<id>-fast` is the fast model of the variant `<id>` (Flash-Next
  /// `--fast-experts`): a different model with its own outputs and cache
  /// identity, unlike the flex tier. Its flex alias is `<id>-fast-flex`.
  static constexpr std::string_view kFastSuffix = "-fast";
  /// A public model id `<variant>[-fast][-flex]` split into the variant it
  /// names (model_id() or a steering adapter) and its suffixes. `fast_ids`:
  /// whether this process serves fast models (otherwise `-fast` is part of
  /// the name). Serve rejects adapter names that would make this ambiguous.
  struct ModelIdParts {
    std::string_view variant;
    bool fast{false};
    bool flex{false};
  };
  [[nodiscard]] static ModelIdParts SplitModelId(std::string_view id,
                                                 bool fast_ids) noexcept {
    ModelIdParts parts{.variant = id};
    if (parts.variant.ends_with(kFlexSuffix)) {
      parts.variant.remove_suffix(kFlexSuffix.size());
      parts.flex = true;
    }
    if (fast_ids && parts.variant.ends_with(kFastSuffix)) {
      parts.variant.remove_suffix(kFastSuffix.size());
      parts.fast = true;
    }
    return parts;
  }
  /// The served id (one of model_ids()) that the flex alias `id` names, or
  /// nullopt when `id` is not a flex alias of this process.
  [[nodiscard]] std::optional<std::string> flex_variant(
      std::string_view id) const {
    if (!id.ends_with(kFlexSuffix))
      return std::nullopt;
    std::string variant(id.substr(0, id.size() - kFlexSuffix.size()));
    const auto ids = model_ids();
    if (std::ranges::find(ids, variant) == ids.end())
      return std::nullopt;
    return variant;
  }
  [[nodiscard]] bool is_flex_model(std::string_view id) const {
    return flex_variant(id).has_value();
  }
  [[nodiscard]] bool serves_model(std::string_view id) const {
    const auto ids = model_ids();
    return std::ranges::find(ids, id) != ids.end() || is_flex_model(id);
  }
  [[nodiscard]] virtual bool ready() const = 0;
  [[nodiscard]] virtual SamplingDefaults sampling_defaults() const {
    return {};
  }
  [[nodiscard]] virtual ReasoningOptions reasoning_defaults() const {
    return {};
  }
  [[nodiscard]] virtual MetricsSnapshot metrics_snapshot() const { return {}; }
  /// Stops admitting requests; in-flight requests finish. Irreversible until
  /// restart. Returns false when the backend cannot drain.
  virtual bool drain() { return false; }
  /// Chat protocol of the served model's template family: the HTTP layer
  /// parses generated text with its output parser.
  [[nodiscard]] virtual const ChatProtocol& chat_protocol() const = 0;

  virtual Result complete(
      std::string_view prompt, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled = {},
      const TokenCallback& on_token = {},
      std::string_view client_id = "anonymous",
      const std::vector<std::string>& stop_sequences = {}) = 0;

  virtual Result chat(const ChatRequest& request, std::size_t max_tokens,
                      const sampling::SamplingConfig& sampling,
                      const CancellationCheck& is_cancelled = {},
                      const TokenCallback& on_token = {}) = 0;

  Result complete(std::string_view prompt, std::size_t max_tokens,
                  float temperature, const CancellationCheck& is_cancelled = {},
                  const TokenCallback& on_token = {},
                  const std::vector<std::string>& stop_sequences = {}) {
    sampling::SamplingConfig config;
    config.temperature = temperature;
    return complete(prompt, max_tokens, config, is_cancelled, on_token,
                    "anonymous", stop_sequences);
  }

  Result chat(const ChatRequest& request, std::size_t max_tokens,
              float temperature, const CancellationCheck& is_cancelled = {},
              const TokenCallback& on_token = {}) {
    sampling::SamplingConfig config;
    config.temperature = temperature;
    return chat(request, max_tokens, config, is_cancelled, on_token);
  }

  /// Reserves admission before a streaming response commits successful headers.
  ///
  /// Backends with an asynchronous scheduler override this. The default
  /// request defers the existing blocking chat call until Wait().
  virtual std::shared_ptr<GenerationRequest> start_chat(
      const ChatRequest& request, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled = {}, bool stream_output = false);

  std::shared_ptr<GenerationRequest> start_chat(
      const ChatRequest& request, std::size_t max_tokens, float temperature,
      const CancellationCheck& is_cancelled = {}, bool stream_output = false) {
    sampling::SamplingConfig config;
    config.temperature = temperature;
    return start_chat(request, max_tokens, config, is_cancelled, stream_output);
  }

  [[nodiscard]] virtual std::size_t count_tokens(
      std::string_view text) const = 0;
};

inline std::shared_ptr<TextGenerationBackend::GenerationRequest>
TextGenerationBackend::start_chat(const ChatRequest& request,
                                  std::size_t max_tokens,
                                  const sampling::SamplingConfig& sampling,
                                  const CancellationCheck& is_cancelled,
                                  bool stream_output) {
  (void)stream_output;
  class DeferredGenerationRequest final : public GenerationRequest {
  public:
    DeferredGenerationRequest(TextGenerationBackend& backend,
                              ChatRequest chat_request, std::size_t token_limit,
                              sampling::SamplingConfig sampling_config,
                              CancellationCheck external_cancellation)
        : backend_(backend),
          request_(std::move(chat_request)),
          max_tokens_(token_limit),
          sampling_(sampling_config),
          external_cancellation_(std::move(external_cancellation)) {}

    Result Wait(const TokenCallback& on_token) override {
      if (waited_.exchange(true, std::memory_order_acq_rel)) {
        throw std::logic_error("generation request was already consumed");
      }
      return backend_.chat(
          request_, max_tokens_, sampling_,
          [this] {
            return cancelled_.load(std::memory_order_acquire) ||
                   (external_cancellation_ && external_cancellation_());
          },
          on_token);
    }

    void Cancel() noexcept override {
      cancelled_.store(true, std::memory_order_release);
    }

  private:
    TextGenerationBackend& backend_;
    ChatRequest request_;
    std::size_t max_tokens_;
    sampling::SamplingConfig sampling_;
    CancellationCheck external_cancellation_;
    std::atomic<bool> waited_{false};
    std::atomic<bool> cancelled_{false};
  };

  return std::make_shared<DeferredGenerationRequest>(*this, request, max_tokens,
                                                     sampling, is_cancelled);
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_TEXT_GENERATION_BACKEND_HPP_

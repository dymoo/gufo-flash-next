#ifndef GUFO_SERVER_TEXT_GENERATION_SCHEDULER_HPP_
#define GUFO_SERVER_TEXT_GENERATION_SCHEDULER_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/cli/serve/text_model_runner.hpp"

namespace gufo::server {

enum class TextRequestPhase : std::uint8_t {
  kQueued,
  kAdmitted,
  kPrefilling,
  kDecodeReady,
  kDecoding,
  kTerminal,
};

inline constexpr std::size_t kDefaultDecodeActivePrefillTokens = 512;
inline constexpr std::chrono::milliseconds kDefaultMaxItl{1500};
inline constexpr std::size_t kDefaultMaxOutputBytes =
    static_cast<std::size_t>(1024) * 1024;
inline constexpr std::size_t kDefaultMaxBufferedOutputBytes =
    static_cast<std::size_t>(64) * 1024;
inline constexpr std::size_t kDefaultMaxBufferedOutputBytesTotal =
    static_cast<std::size_t>(256) * 1024;

/// Prefill while other requests decode (`--prefill-chunk`).
///
/// Runners that fuse prefill (TextRunnerCapabilities::fused_prefill_rows,
/// Flash-Next) pack prompt chunks of several requests into every decode
/// step and, with nothing decoding, into one prefill step: all chunks share
/// the step's weight and expert reads, and decoders keep producing a token
/// per step. Other runners alternate one bounded chunk with a decode step.
struct TextPrefillPolicy {
  /// The chunk size of an alternating runner, and the starting prompt rows
  /// of a fused decode step. With nothing decoding, a step carries the
  /// runner's fused_prefill_rows.
  std::size_t decode_active_tokens{kDefaultDecodeActivePrefillTokens};
  /// Target duration of a fused decode step, i.e. the inter-token latency
  /// decoders see while prompts wait (`--max-itl-ms`). Prompt rows per
  /// fused step grow by a quarter while steps finish under 90% of it and
  /// shrink by a quarter above it, within [decode_active_tokens / 4,
  /// fused_prefill_rows]: a step streams every expert once up to about
  /// 1-2K rows, so packing more prompt rows is nearly free until the step
  /// turns compute-bound. Zero keeps the fixed budget below.
  std::chrono::milliseconds max_itl{kDefaultMaxItl};
  /// Prompts with the fewest remaining tokens go first, each to completion
  /// while rows remain, so short tool results are not queued behind a long
  /// prompt and several share one step. A request passed over this many
  /// steps in a row goes first (aging), so long prompts still progress.
  std::size_t max_prefill_skips{32};
  static constexpr std::size_t kFusedDecodeRows = 4;
};

struct TextSchedulerPolicy {
  std::size_t max_pending_requests{16};
  std::size_t max_pending_requests_per_client{4};
  /// Sessions flex requests may hold at once (`--flex-sessions`); 0 rejects
  /// every flex request.
  std::size_t flex_sessions{2};
  /// `--admission refuse`: a default request that finds every session taken
  /// gets kCapacityExhausted at once instead of waiting in the pending queue.
  bool refuse_when_full{false};
  std::size_t max_output_bytes_per_request{kDefaultMaxOutputBytes};
  std::size_t max_buffered_output_bytes_per_request{
      kDefaultMaxBufferedOutputBytes};
  std::size_t max_buffered_output_bytes_total{
      kDefaultMaxBufferedOutputBytesTotal};
  std::chrono::milliseconds request_timeout{0};
  bool log_progress{false};
};

/// Best-effort session/pool gauges for /metrics; see
/// TextGenerationScheduler::metrics_snapshot().
struct TextSchedulerMetrics {
  std::size_t queued{0};
  std::size_t queued_flex{0};
  std::size_t flex_resident{0};
  std::size_t default_prefilling{0};
  bool draining{false};
  std::size_t resident{0};
  std::size_t active{0};
  std::size_t kv_pool_bytes_used{0};
  std::size_t kv_pool_bytes_limit{0};
  std::size_t host_snapshot_bytes{0};
  std::size_t host_snapshot_bytes_limit{0};
  std::size_t disk_cache_bytes{0};
  std::size_t disk_cache_bytes_limit{0};
  std::size_t disk_cache_entries{0};
};

/// Single-owner scheduler for opaque text-model runner states.
///
/// Submitters never execute model code. One scheduler thread owns admission,
/// runner leases, prefill/decode work units, cancellation, and reclamation.
class TextGenerationScheduler {
public:
  using Clock = std::chrono::steady_clock;
  using Result = TextGenerationBackend::Result;
  using CancellationCheck = TextGenerationBackend::CancellationCheck;
  using TokenCallback = TextGenerationBackend::TokenCallback;

  struct RequestMetadata {
    std::string client_id{"anonymous"};
    std::optional<Clock::time_point> deadline;
    Clock::time_point request_start{Clock::now()};
    std::shared_ptr<const TextPromptContext> prompt_context;
    bool cache_prompt{true};
    std::size_t cache_prefix_tokens{0};
    std::vector<std::string> stop_sequences;
    ServiceTier service_tier{ServiceTier::kDefault};
    /// Reject with kQueueFull instead of queueing when no session is free.
    bool no_queue{false};
    /// Queue even when the policy refuses when full.
    bool queue{false};
  };

  class Request {
  public:
    Request();
    ~Request();

    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;
    Request(Request&&) noexcept;
    Request& operator=(Request&&) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] std::uint64_t id() const noexcept;
    [[nodiscard]] TextRequestPhase phase() const noexcept;

    /// Consumes queued output pieces on the calling thread and waits for the
    /// scheduler-owned request to become terminal.
    Result Wait(const TokenCallback& on_token = {});
    void Cancel() noexcept;

  private:
    friend class TextGenerationScheduler;
    struct Impl;

    explicit Request(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
  };

  explicit TextGenerationScheduler(std::shared_ptr<TextRunnerPool> runner_pool,
                                   TextPrefillPolicy prefill_policy = {},
                                   TextSchedulerPolicy scheduler_policy = {});
  ~TextGenerationScheduler();

  TextGenerationScheduler(const TextGenerationScheduler&) = delete;
  TextGenerationScheduler& operator=(const TextGenerationScheduler&) = delete;
  TextGenerationScheduler(TextGenerationScheduler&&) = delete;
  TextGenerationScheduler& operator=(TextGenerationScheduler&&) = delete;

  [[nodiscard]] const TextModelRunner& runner() const noexcept;
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t buffered_output_bytes() const noexcept;
  [[nodiscard]] std::size_t max_buffered_output_bytes() const noexcept;
  /// Snapshot for /metrics gauges; cheap enough to call once per scrape.
  [[nodiscard]] TextSchedulerMetrics metrics_snapshot() const;
  [[nodiscard]] const TextSchedulerPolicy& policy() const noexcept;
  /// Rejects new requests with kDraining; admitted and queued ones finish.
  void Drain() noexcept;
  /// Throws the refusal Submit would raise now for a request that may not
  /// wait (draining, flex, no-queue, refuse mode), so a caller can skip
  /// prompt preparation such as image encoding. Submit still decides.
  void CheckAdmission(const RequestMetadata& metadata) const;

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens,
                               const sampling::SamplingConfig& sampling,
                               const CancellationCheck& is_cancelled = {},
                               bool publish_token_pieces = false);

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens,
                               const sampling::SamplingConfig& sampling,
                               const CancellationCheck& is_cancelled,
                               bool publish_token_pieces,
                               RequestMetadata metadata);

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens, float temperature,
                               const CancellationCheck& is_cancelled = {},
                               bool publish_token_pieces = false);

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens, float temperature,
                               const CancellationCheck& is_cancelled,
                               bool publish_token_pieces,
                               RequestMetadata metadata);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

using TextRequestMetadata = TextGenerationScheduler::RequestMetadata;

}  // namespace gufo::server

#endif  // GUFO_SERVER_TEXT_GENERATION_SCHEDULER_HPP_

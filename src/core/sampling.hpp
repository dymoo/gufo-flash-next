#ifndef GUFO_CORE_SAMPLING_HPP_
#define GUFO_CORE_SAMPLING_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace gufo::sampling {

using TokenId = std::uint32_t;

/// Request-owned restriction of the next token (grammar-constrained
/// decoding). The serving pool reports each committed token through Accept;
/// SamplerState::Sample masks the rows of tokens the constraint forbids.
/// Speculative paths never see constrained requests: the scheduler decodes
/// them autoregressively.
class TokenConstraint {
public:
  TokenConstraint() = default;
  virtual ~TokenConstraint() = default;
  TokenConstraint(const TokenConstraint&) = delete;
  TokenConstraint& operator=(const TokenConstraint&) = delete;
  TokenConstraint(TokenConstraint&&) = delete;
  TokenConstraint& operator=(TokenConstraint&&) = delete;

  /// Whether the next token is restricted (false while reasoning runs free).
  [[nodiscard]] virtual bool Restricts() const = 0;
  /// Non-blocking: the next token's mask is ready, or none is needed.
  [[nodiscard]] virtual bool Ready() const = 0;
  /// Sets the forbidden entries of a full-vocabulary row to -infinity,
  /// waiting for the mask when it is still being computed.
  virtual void Mask(std::span<float> logits) = 0;
  /// Commits one generated token; throws std::runtime_error when the token
  /// violates the constraint.
  virtual void Accept(TokenId token) = 0;
};

/// Model-independent controls for selecting a token from a logit row.
///
/// Filters run in this order: penalties, temperature, top-k, top-p, min-p.
/// The defaults preserve Gufo's existing greedy decoding behavior.
struct SamplingConfig {
  float temperature{0.0F};
  std::int32_t top_k{0};
  float top_p{1.0F};
  float min_p{0.0F};
  std::size_t min_keep{0};
  std::int64_t seed{-1};
  float repeat_penalty{1.0F};
  std::size_t repeat_last_n{64};
  float frequency_penalty{0.0F};
  float presence_penalty{0.0F};
  /// Evaluation probe: report each selected token's log-probability under the
  /// unmodified model distribution plus this many alternatives (0-20).
  /// -1 disables reporting.
  std::int32_t top_logprobs{-1};
  /// Teacher forcing: Sample returns these tokens in order before drawing.
  std::vector<TokenId> forced_tokens;
  /// Grammar-constrained decoding; shared by every copy of this request's
  /// configuration and advanced only by the serving pool.
  std::shared_ptr<TokenConstraint> constraint;

  void Validate() const;

  [[nodiscard]] bool penalties_enabled() const noexcept;
  [[nodiscard]] bool uses_random_sampling() const noexcept;
  [[nodiscard]] bool can_use_unmodified_argmax() const noexcept;
  /// Probing requests need one Sample call per emitted token (no drafts).
  [[nodiscard]] bool probing() const noexcept {
    return top_logprobs >= 0 || !forced_tokens.empty();
  }
  /// Requests that sample every token on the host from full logits: probes
  /// and constrained requests decode autoregressively, without drafts.
  [[nodiscard]] bool host_sampled() const noexcept {
    return probing() || constraint != nullptr;
  }
};

/// Log-softmax of the raw logits (before penalties and temperature) for one
/// selected token, with the most likely alternatives in descending order.
struct TokenLogprobs {
  TokenId token{0};
  double logprob{0.0};
  std::vector<std::pair<TokenId, double>> top;
};

[[nodiscard]] TokenLogprobs ComputeLogprobs(std::span<const float> logits,
                                            TokenId token, std::size_t top);

/// Unique token counts: repetition window and full generated response.
struct TokenPenalty {
  TokenId token{0};
  std::uint32_t generated_count{0};
  std::uint32_t repeated{0};
};

struct Probability {
  TokenId token{0};
  double value{0.0};
};

/// A normalized, post-filter sampling distribution.
class SamplingDistribution {
public:
  SamplingDistribution() = default;
  explicit SamplingDistribution(std::vector<Probability> entries);

  [[nodiscard]] std::span<const Probability> entries() const noexcept;
  [[nodiscard]] TokenId best_token() const;
  [[nodiscard]] double probability(TokenId token) const noexcept;
  [[nodiscard]] TokenId Sample(std::uint64_t* rng_state) const;
  [[nodiscard]] TokenId SampleResidual(
      std::span<const TokenId> candidate_ids,
      std::span<const float> candidate_probabilities,
      std::uint64_t* rng_state) const;

private:
  friend class SamplerState;
  // Entries generated internally have unique IDs; retain their stable order.
  SamplingDistribution(std::vector<Probability> entries, double total);
  std::vector<Probability> entries_;
};

/// Builds the exact distribution from prompt and request-generated tokens.
[[nodiscard]] SamplingDistribution BuildDistribution(
    std::span<const float> logits, const SamplingConfig& config,
    std::span<const TokenId> prompt_tokens = {},
    std::span<const TokenId> generated_tokens = {});

/// Request-owned controls, repetition window, full generated counts and RNG.
///
/// Call Accept only for committed tokens. The type is copyable so speculative
/// decode can work on a tentative state and discard it on rollback.
///
/// The RNG is counter-based: each committed position restarts the draw stream
/// at H(seed, position), so a draw made or skipped at one position (batch
/// shape, draft length, a one-entry distribution) never shifts later ones.
class SamplerState {
public:
  explicit SamplerState(SamplingConfig config = {},
                        std::span<const TokenId> initial_history = {});
  ~SamplerState() = default;
  SamplerState(const SamplerState& other);
  SamplerState& operator=(const SamplerState& other);
  SamplerState(SamplerState&&) noexcept = default;
  SamplerState& operator=(SamplerState&&) noexcept = default;

  [[nodiscard]] const SamplingConfig& config() const noexcept;
  [[nodiscard]] std::span<const TokenId> history() const noexcept;
  [[nodiscard]] std::span<const TokenPenalty> penalties() const noexcept {
    return penalty_counts_;
  }
  [[nodiscard]] std::uint64_t rng_state() const noexcept;
  [[nodiscard]] std::uint64_t* mutable_rng_state() noexcept;
  void SetRngState(std::uint64_t state) noexcept;
  /// Publish RNG and any deferred draw without accepting tentative history.
  /// The RNG is copied only from a state at this committed position: draws
  /// of a copy that accepted further tokens belong to its later position,
  /// which this sampler keys afresh once it accepts those tokens.
  void CopyDrawStateFrom(const SamplerState& other) noexcept;
  struct DrawState {
    std::uint64_t rng;
    std::optional<TokenId> pending;
  };
  [[nodiscard]] DrawState SaveDrawState() const noexcept {
    return {rng_state_, pending_sample_};
  }
  void RestoreDrawState(DrawState state) noexcept {
    rng_state_ = state.rng;
    pending_sample_ = state.pending;
  }

  /// Starts a request: tokens initialize repetition only; generated counts
  /// reset.
  void ResetHistory(std::span<const TokenId> tokens);
  void Accept(TokenId token);
  void Accept(std::span<const TokenId> tokens);

  [[nodiscard]] SamplingDistribution Distribution(
      std::span<const float> logits) const;
  /// Compact logits use token_ids for penalties; returned IDs index logits.
  [[nodiscard]] SamplingDistribution Distribution(
      std::span<const float> logits, std::span<const TokenId> token_ids) const;
  /// Retain a speculative residual draw for the next Sample call. Copies
  /// preserve this draw along with the RNG; resetting history discards it.
  void DeferSample(TokenId token);
  /// Returns the next forced token, else draws. Records TakeLogprobs() when
  /// config().top_logprobs >= 0.
  [[nodiscard]] TokenId Sample(std::span<const float> logits);
  [[nodiscard]] std::optional<TokenLogprobs> TakeLogprobs() noexcept {
    return std::exchange(logprobs_, std::nullopt);
  }
  [[nodiscard]] TokenId SampleResidual(
      std::span<const float> target_logits,
      std::span<const TokenId> candidate_ids,
      std::span<const float> candidate_probabilities);
  [[nodiscard]] double Uniform();

private:
  void TrimHistory();
  void RebuildPenaltyCounts();
  void KeyRng() noexcept;
  [[nodiscard]] double AdjustedLogit(TokenId token, float logit) const noexcept;
  [[nodiscard]] TokenId SampleGreedy(std::span<const float> logits) const;
  [[nodiscard]] TokenId Draw(std::span<const float> logits);
  [[nodiscard]] SamplingDistribution LinearDistribution(
      std::span<const float> logits) const;
  void PrepareSelected(std::span<const float> logits);

  SamplingConfig config_;
  std::vector<TokenId> history_;
  std::vector<TokenPenalty> penalty_counts_;
  std::vector<Probability> candidate_scratch_;
  std::uint64_t seed_key_{0};
  std::uint64_t position_{0};  ///< tokens accepted since the request began
  std::uint64_t rng_state_{0};
  std::optional<TokenId> pending_sample_;
  std::size_t forced_index_{0};
  std::optional<TokenLogprobs> logprobs_;
};

[[nodiscard]] std::uint64_t NextRandom(std::uint64_t* state);
[[nodiscard]] double Uniform(std::uint64_t* state);

// Compatibility helpers for callers that have not yet adopted SamplerState.
[[nodiscard]] TokenId SampleLogits(std::span<const float> logits,
                                   float temperature, std::uint64_t* rng_state);
[[nodiscard]] double TokenProbability(std::span<const float> logits,
                                      float temperature, TokenId token);
[[nodiscard]] TokenId SampleResidual(
    std::span<const float> target_logits, float temperature,
    std::span<const TokenId> candidate_ids,
    std::span<const float> candidate_probabilities, std::uint64_t* rng_state);

}  // namespace gufo::sampling

#endif  // GUFO_CORE_SAMPLING_HPP_

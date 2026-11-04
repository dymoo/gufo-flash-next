#ifndef GUFO_CORE_GRAMMAR_GRAMMAR_HPP_
#define GUFO_CORE_GRAMMAR_GRAMMAR_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/core/sampling.hpp"

/// Grammar-constrained decoding over the vendored xgrammar core
/// (src/vendor/xgrammar). One Engine serves one model vocabulary: it compiles
/// JSON schemas and structural tags (cached under a byte cap, with a time
/// limit) and creates per-request sampling::TokenConstraint objects whose
/// next-token masks are computed on worker threads while the model runs.
namespace gufo::grammar {

using sampling::TokenId;

/// The vocabulary as the grammar sees it. Each entry holds a token's raw
/// output bytes; an empty entry is a control token the grammar never emits.
struct Vocabulary {
  std::vector<std::string> tokens;
  std::vector<std::int32_t> stop_tokens;
  /// Width of the model's logit rows (at least tokens.size()).
  std::size_t logits_width{0};
};

/// What a constraint enforces once it applies.
struct GrammarSource {
  enum class Kind : std::uint8_t {
    kJsonSchema,     ///< `text` is a JSON Schema document
    kStructuralTag,  ///< `text` is an xgrammar structural tag
  };
  Kind kind{Kind::kJsonSchema};
  std::string text;
};

/// How a template family's reasoning surrounds the constrained answer.
struct Gating {
  enum class Start : std::uint8_t {
    kAnswer,     ///< the grammar applies from the first token
    kReasoning,  ///< free reasoning until `reasoning_end`
    kEither,     ///< the model may open `reasoning_start` or answer at once
  };
  Start start{Start::kAnswer};
  /// Tokens that close reasoning, e.g. "</think>".
  std::vector<TokenId> reasoning_end;
  /// Single token that opens reasoning (kEither only).
  std::optional<TokenId> reasoning_start;
  /// Tokens the template writes between reasoning and the answer.
  std::vector<TokenId> separator;
};

struct Limits {
  /// xgrammar's compiled-grammar cache; its own default is unbounded.
  std::size_t cache_bytes{std::size_t{512} * 1024 * 1024};
  std::chrono::milliseconds compile_timeout{5000};
  /// Grammar source (schema or structural tag) size limit.
  std::size_t max_source_bytes{std::size_t{512} * 1024};
  /// Concurrent compilations (cache hits included); further requests wait
  /// in a bounded queue, each within its own compile timeout.
  unsigned compile_workers{2};
  unsigned compile_queue{1024};
  /// Threads computing per-token masks for all constrained requests.
  unsigned mask_workers{4};
};

/// Process-wide limits for engines created afterwards (`gufo serve` flags).
void SetDefaultLimits(const Limits& limits);
[[nodiscard]] Limits DefaultLimits();

/// A request the grammar cannot serve. Every case is a client-visible 400:
/// gateways fail over instead of retrying.
class GrammarError final : public std::invalid_argument {
public:
  GrammarError(std::string code, const std::string& message)
      : std::invalid_argument(message), code_(std::move(code)) {}
  [[nodiscard]] const std::string& code() const noexcept { return code_; }

private:
  std::string code_;
};

class Engine {
public:
  /// Starts building the vocabulary tables on a background thread.
  explicit Engine(Vocabulary vocabulary, Limits limits = DefaultLimits());
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine(Engine&&) = delete;
  Engine& operator=(Engine&&) = delete;

  /// Compiles `source` (or reuses the cached grammar) within the compile
  /// limit and returns a fresh request constraint. Throws GrammarError.
  [[nodiscard]] std::shared_ptr<sampling::TokenConstraint> NewConstraint(
      const GrammarSource& source, const Gating& gating) const;

  /// Bytes held by the compiled-grammar cache.
  [[nodiscard]] std::size_t CacheBytes() const;
  [[nodiscard]] const Limits& limits() const noexcept;

  struct Impl;

private:
  std::shared_ptr<Impl> impl_;
};

}  // namespace gufo::grammar

#endif  // GUFO_CORE_GRAMMAR_GRAMMAR_HPP_

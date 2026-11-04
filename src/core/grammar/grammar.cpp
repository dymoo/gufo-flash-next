#include "src/core/grammar/grammar.hpp"

#include <dlpack/dlpack.h>
#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <span>
#include <thread>

// Optimization inspiration: DeepSeek DeepInfer (Apache-2.0) — asynchronous
// grammar compilation with a time limit, CPU token masks prepared while the
// model computes the next row, and trigger-token gating so reasoning runs
// unconstrained. No DeepInfer code is copied here.

namespace gufo::grammar {
namespace {

/// Consecutive JSON whitespace characters a grammar admits. Unbounded
/// whitespace lets a weak sampling path loop on newlines forever.
constexpr int kMaxJsonWhitespace = 32;
/// xgrammar's own threads for one compilation's token-mask preprocessing.
constexpr int kCompileThreads = 2;
constexpr std::size_t kMaxErrorBytes = 512;

std::mutex& LimitsMutex() {
  static std::mutex mutex;
  return mutex;
}

Limits& ProcessLimits() {
  static Limits limits;
  return limits;
}

/// Fixed worker threads draining a FIFO of jobs.
class WorkQueue {
public:
  WorkQueue(unsigned workers, std::size_t capacity) : capacity_(capacity) {
    workers = std::max(workers, 1U);
    threads_.reserve(workers);
    for (unsigned i = 0; i < workers; ++i)
      threads_.emplace_back([this] { Run(); });
  }

  ~WorkQueue() {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    for (auto& thread : threads_)
      thread.join();
  }

  WorkQueue(const WorkQueue&) = delete;
  WorkQueue& operator=(const WorkQueue&) = delete;
  WorkQueue(WorkQueue&&) = delete;
  WorkQueue& operator=(WorkQueue&&) = delete;

  /// False when the queue is full or stopping; the job is not run.
  bool Post(std::function<void()> job) {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_ || jobs_.size() >= capacity_)
        return false;
      jobs_.push_back(std::move(job));
    }
    ready_.notify_one();
    return true;
  }

private:
  void Run() {
    for (;;) {
      std::function<void()> job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
        if (jobs_.empty())
          return;
        job = std::move(jobs_.front());
        jobs_.pop_front();
      }
      job();  // Jobs report their own failures through promises.
    }
  }

  std::size_t capacity_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<std::function<void()>> jobs_;
  bool stopping_{false};
  std::vector<std::thread> threads_;
};

std::string Truncated(std::string message) {
  if (message.size() > kMaxErrorBytes) {
    message.resize(kMaxErrorBytes);
    message += "...";
  }
  return message;
}

/// Vocabulary tables and the compiler (and its cache). Immutable once built
/// except for the compiler's internally synchronized cache.
struct Core {
  Core(const Vocabulary& vocabulary, std::size_t cache_bytes)
      : vocab_size(static_cast<int>(
            std::max(vocabulary.logits_width, vocabulary.tokens.size()))),
        info(vocabulary.tokens, xgrammar::VocabType::RAW, vocab_size,
             vocabulary.stop_tokens, false),
        compiler(info, kCompileThreads, true,
                 static_cast<std::int64_t>(std::min<std::size_t>(
                     cache_bytes, std::numeric_limits<std::int64_t>::max()))) {}

  int vocab_size;
  xgrammar::TokenizerInfo info;
  xgrammar::GrammarCompiler compiler;
};

/// Matcher and its mask buffer: shared with the mask job that fills it.
struct MatcherState {
  MatcherState(const xgrammar::CompiledGrammar& grammar, int vocab_size)
      : matcher(grammar),
        bitmask(static_cast<std::size_t>(xgrammar::GetBitmaskSize(vocab_size)),
                -1) {
    shape[0] = static_cast<std::int64_t>(bitmask.size());
    tensor.data = bitmask.data();
    tensor.device = DLDevice{kDLCPU, 0};
    tensor.ndim = 1;
    tensor.dtype = xgrammar::GetBitmaskDLType();
    tensor.shape = shape.data();
    tensor.strides = nullptr;
    tensor.byte_offset = 0;
  }

  void Fill() { restricted = matcher.FillNextTokenBitmask(&tensor); }

  xgrammar::GrammarMatcher matcher;
  std::vector<std::int32_t> bitmask;
  std::array<std::int64_t, 1> shape{};
  DLTensor tensor{};
  bool restricted{true};
};

}  // namespace

struct Engine::Impl {
  Impl(Vocabulary vocabulary, Limits engine_limits)
      : limits(engine_limits),
        compile_queue(engine_limits.compile_workers,
                      engine_limits.compile_queue),
        mask_queue(engine_limits.mask_workers,
                   std::numeric_limits<std::size_t>::max()) {
    const std::size_t cache_bytes = limits.cache_bytes;
    core = std::async(std::launch::async, [vocabulary = std::move(vocabulary),
                                           cache_bytes] {
             return std::make_shared<Core>(vocabulary, cache_bytes);
           }).share();
  }

  [[nodiscard]] std::shared_ptr<Core> GetCore() const {
    try {
      return core.get();
    } catch (const std::exception& exception) {
      throw GrammarError(
          "grammar_unavailable",
          Truncated(std::string("structured outputs are unavailable: ") +
                    exception.what()));
    }
  }

  Limits limits;
  std::shared_future<std::shared_ptr<Core>> core;
  mutable WorkQueue compile_queue;
  mutable WorkQueue mask_queue;
};

namespace {

/// One request's constraint: optional free reasoning, forced separator
/// tokens, then the grammar. Masks for the next token are computed on the
/// engine's mask workers as soon as a token is accepted, overlapping the
/// model step that produces the next logit row.
class Constraint final : public sampling::TokenConstraint {
public:
  Constraint(std::shared_ptr<Engine::Impl> engine,
             const xgrammar::CompiledGrammar& grammar, int vocab_size,
             Gating gating)
      : engine_(std::move(engine)),
        state_(std::make_shared<MatcherState>(grammar, vocab_size)),
        gating_(std::move(gating)),
        failure_(Failure(gating_.reasoning_end)) {
    switch (gating_.start) {
      case Gating::Start::kAnswer:
        phase_ = Phase::kAnswer;
        Schedule();
        break;
      case Gating::Start::kReasoning:
        phase_ =
            gating_.reasoning_end.empty() ? Phase::kAnswer : Phase::kReasoning;
        if (phase_ == Phase::kAnswer)
          Schedule();
        break;
      case Gating::Start::kEither:
        phase_ = Phase::kEither;
        Schedule();
        break;
    }
  }

  ~Constraint() override {
    // The job owns a reference to the matcher; nothing to join.
  }

  Constraint(const Constraint&) = delete;
  Constraint& operator=(const Constraint&) = delete;
  Constraint(Constraint&&) = delete;
  Constraint& operator=(Constraint&&) = delete;

  [[nodiscard]] bool Restricts() const override {
    return phase_ != Phase::kReasoning;
  }

  [[nodiscard]] bool Ready() const override {
    return !job_.valid() ||
           job_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
  }

  void Mask(std::span<float> logits) override {
    constexpr float kForbidden = -std::numeric_limits<float>::infinity();
    switch (phase_) {
      case Phase::kReasoning:
        return;
      case Phase::kSeparator: {
        const TokenId allowed = gating_.separator[separator_index_];
        for (std::size_t token = 0; token < logits.size(); ++token)
          if (token != allowed)
            logits[token] = kForbidden;
        return;
      }
      case Phase::kEither:
      case Phase::kAnswer:
        break;
    }
    Wait();
    std::optional<float> reasoning_logit;
    const auto start = gating_.reasoning_start;
    if (phase_ == Phase::kEither && start && *start < logits.size())
      reasoning_logit = logits[*start];
    const auto& words = state_->bitmask;
    const std::size_t covered = words.size() * 32;
    for (std::size_t token = covered; token < logits.size(); ++token)
      logits[token] = kForbidden;
    if (state_->restricted) {
      for (std::size_t word = 0; word < words.size(); ++word) {
        const auto bits = static_cast<std::uint32_t>(words[word]);
        if (bits == std::numeric_limits<std::uint32_t>::max())
          continue;
        const std::size_t base = word * 32;
        const std::size_t end = std::min(base + 32, logits.size());
        for (std::size_t token = base; token < end; ++token)
          if (((bits >> (token - base)) & 1U) == 0)
            logits[token] = kForbidden;
      }
    }
    if (reasoning_logit)
      logits[*start] = *reasoning_logit;
  }

  void Accept(TokenId token) override {
    switch (phase_) {
      case Phase::kReasoning:
        AdvanceReasoning(token);
        return;
      case Phase::kSeparator:
        if (token != gating_.separator[separator_index_])
          throw std::runtime_error(
              "generated token breaks the reasoning separator");
        if (++separator_index_ == gating_.separator.size()) {
          phase_ = Phase::kAnswer;
          Schedule();
        }
        return;
      case Phase::kEither:
        Wait();
        if (gating_.reasoning_start && token == *gating_.reasoning_start) {
          phase_ = Phase::kReasoning;
          return;
        }
        phase_ = Phase::kAnswer;
        break;
      case Phase::kAnswer:
        Wait();
        break;
    }
    if (!state_->matcher.AcceptToken(static_cast<std::int32_t>(token)))
      throw std::runtime_error("generated token violates the output grammar");
    Schedule();
  }

private:
  enum class Phase : std::uint8_t {
    kEither,
    kReasoning,
    kSeparator,
    kAnswer,
  };

  static std::vector<std::size_t> Failure(std::span<const TokenId> pattern) {
    std::vector<std::size_t> failure(pattern.size(), 0);
    std::size_t matched = 0;
    for (std::size_t i = 1; i < pattern.size(); ++i) {
      while (matched > 0 && pattern[i] != pattern[matched])
        matched = failure[matched - 1];
      if (pattern[i] == pattern[matched])
        ++matched;
      failure[i] = matched;
    }
    return failure;
  }

  void AdvanceReasoning(TokenId token) {
    const auto& end = gating_.reasoning_end;
    while (matched_ > 0 && token != end[matched_])
      matched_ = failure_[matched_ - 1];
    if (token == end[matched_])
      ++matched_;
    if (matched_ < end.size())
      return;
    if (gating_.separator.empty()) {
      phase_ = Phase::kAnswer;
      Schedule();
    } else {
      phase_ = Phase::kSeparator;
      separator_index_ = 0;
    }
  }

  void Schedule() {
    auto task = std::make_shared<std::packaged_task<void()>>(
        [state = state_] { state->Fill(); });
    job_ = task->get_future();
    if (!engine_->mask_queue.Post([task] { (*task)(); }))
      (*task)();
  }

  void Wait() {
    if (job_.valid())
      job_.get();
  }

  std::shared_ptr<Engine::Impl> engine_;
  std::shared_ptr<MatcherState> state_;
  Gating gating_;
  std::vector<std::size_t> failure_;
  std::atomic<Phase> phase_{Phase::kAnswer};
  std::size_t matched_{0};
  std::size_t separator_index_{0};
  std::future<void> job_;
};

xgrammar::CompiledGrammar Compile(Core& core, const GrammarSource& source) {
  if (source.kind == GrammarSource::Kind::kStructuralTag)
    return core.compiler.CompileStructuralTag(source.text);
  return core.compiler.CompileJSONSchema(
      source.text, true, std::nullopt, std::nullopt, true, kMaxJsonWhitespace);
}

}  // namespace

void SetDefaultLimits(const Limits& limits) {
  const std::lock_guard<std::mutex> lock(LimitsMutex());
  ProcessLimits() = limits;
}

Limits DefaultLimits() {
  const std::lock_guard<std::mutex> lock(LimitsMutex());
  return ProcessLimits();
}

Engine::Engine(Vocabulary vocabulary, Limits limits)
    : impl_(std::make_shared<Impl>(std::move(vocabulary), limits)) {}

Engine::~Engine() = default;

const Limits& Engine::limits() const noexcept {
  return impl_->limits;
}

std::size_t Engine::CacheBytes() const {
  const auto bytes = impl_->GetCore()->compiler.GetCacheSizeBytes();
  return bytes > 0 ? static_cast<std::size_t>(bytes) : 0;
}

std::shared_ptr<sampling::TokenConstraint> Engine::NewConstraint(
    const GrammarSource& source, const Gating& gating) const {
  if (source.text.size() > impl_->limits.max_source_bytes) {
    throw GrammarError("grammar_too_large",
                       "structured output schema exceeds " +
                           std::to_string(impl_->limits.max_source_bytes) +
                           " bytes");
  }
  const auto core = impl_->GetCore();
  auto promise = std::make_shared<std::promise<xgrammar::CompiledGrammar>>();
  auto compiled = promise->get_future();
  const bool queued =
      impl_->compile_queue.Post([core, source, promise]() mutable {
        try {
          promise->set_value(Compile(*core, source));
        } catch (...) {
          promise->set_exception(std::current_exception());
        }
      });
  if (!queued) {
    throw GrammarError("grammar_busy",
                       "structured output compiler queue is full");
  }
  if (compiled.wait_for(impl_->limits.compile_timeout) !=
      std::future_status::ready) {
    // The compilation finishes in the background and fills the cache.
    throw GrammarError(
        "grammar_timeout",
        "structured output grammar did not compile within " +
            std::to_string(impl_->limits.compile_timeout.count()) + " ms");
  }
  try {
    return std::make_shared<Constraint>(impl_, compiled.get(), core->vocab_size,
                                        gating);
  } catch (const GrammarError&) {
    throw;
  } catch (const std::exception& exception) {
    throw GrammarError("invalid_grammar",
                       Truncated(std::string("unsupported structured output "
                                             "schema: ") +
                                 exception.what()));
  }
}

}  // namespace gufo::grammar

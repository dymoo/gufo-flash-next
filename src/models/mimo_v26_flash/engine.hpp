#ifndef GUFO_MODELS_MIMO_V26_FLASH_ENGINE_HPP_
#define GUFO_MODELS_MIMO_V26_FLASH_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/mimo_v26_flash/weights.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::mimo_v26_flash {

namespace rocm {
class Executor;
class KvState;
}  // namespace rocm

struct ModelOptions {
  std::uint32_t max_context = 32768;
  /// Device bytes for full-attention KV pages; zero takes free memory less a
  /// safety margin.
  std::size_t kv_pool_bytes = 0;
};

class Session;

/// One resident MiMo-V2.6-Flash trunk; sessions share batched steps.
class Model final : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& path, const ModelOptions& options,
      std::string* error = nullptr);
  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::string* error = nullptr);

  [[nodiscard]] const tokenization::QwenTokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] std::string TokenText(std::int32_t token) const;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return options_.max_context;
  }
  /// Prompt rows one step may carry.
  [[nodiscard]] std::uint32_t PrefillCapacity() const noexcept;
  [[nodiscard]] std::string ModelName() const;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  [[nodiscard]] std::size_t KvPoolBytes() const noexcept;
  [[nodiscard]] std::size_t KvPoolUsed() const noexcept;
  [[nodiscard]] const core::GgufReader& reader() const noexcept {
    return *reader_;
  }

private:
  friend class Session;
  Model() = default;

  ModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer_;
  std::unique_ptr<rocm::Executor> executor_;
  std::mutex device_mutex_;
};

/// One conversation. The state advances only by appending: Sync keeps the
/// session when the prompt extends its tokens and otherwise starts over
/// (sliding-window state cannot be truncated); snapshots restore earlier
/// points exactly.
class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error = nullptr);
  [[nodiscard]] bool Evaluate(std::int32_t token, std::string* error = nullptr);

  struct BatchOutcome {
    bool completed{false};
    std::string error;
  };
  struct AdvanceRequest {
    Session* session;
    std::int32_t token;
    BatchOutcome* outcome{nullptr};
  };
  /// Grows `session` towards `prompt` (which must extend its tokens) by at
  /// most `max_tokens` positions.
  struct PrefillRequest {
    Session* session;
    std::span<const std::int32_t> prompt;
    std::size_t max_tokens;
    BatchOutcome* outcome{nullptr};
  };
  /// One device step for every request. Returns false when the step failed;
  /// then every outcome is failed and the sessions involved are reset.
  [[nodiscard]] static bool Step(std::span<const AdvanceRequest> advances,
                                 std::span<const PrefillRequest> prefills,
                                 std::string* error = nullptr);

  [[nodiscard]] std::span<const float> Logits() const noexcept {
    return logits_valid_ ? std::span<const float>(logits_)
                         : std::span<const float>{};
  }
  [[nodiscard]] std::uint32_t Position() const noexcept {
    return static_cast<std::uint32_t>(tokens_.size());
  }
  [[nodiscard]] std::span<const std::int32_t> Tokens() const noexcept {
    return tokens_;
  }
  void Reset();
  [[nodiscard]] std::size_t CommittedBytes() const noexcept;

  static constexpr std::uint32_t kSnapshotVersion = 1;
  [[nodiscard]] std::uint64_t SnapshotBytes() const;
  /// The whole context (tokens, last logits, KV pages, window ring).
  [[nodiscard]] bool SaveSnapshot(std::vector<std::uint8_t>* out,
                                  std::string* error = nullptr) const;
  [[nodiscard]] bool RestoreSnapshot(std::span<const std::uint8_t> payload,
                                     std::string* error = nullptr);

private:
  friend class Model;
  Session(std::shared_ptr<Model> model, std::unique_ptr<rocm::KvState> kv);

  std::shared_ptr<Model> model_;
  std::unique_ptr<rocm::KvState> kv_;
  std::vector<std::int32_t> tokens_;
  std::vector<float> logits_;
  bool logits_valid_{false};
};

}  // namespace gufo::models::mimo_v26_flash

#endif  // GUFO_MODELS_MIMO_V26_FLASH_ENGINE_HPP_

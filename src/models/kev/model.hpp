#ifndef GUFO_MODELS_KEV_MODEL_HPP_
#define GUFO_MODELS_KEV_MODEL_HPP_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/kev/encoding.hpp"
#include "src/models/kev/hip/packed.hpp"

namespace gufo::hip {
class QwenGpuExecutor;
class QwenGpuModel;
}  // namespace gufo::hip

namespace gufo::models::kev {

/// kev.model.PointerHead: z_i = k(h_option_i) . q(h_decide) / sqrt(d) / T.
struct PointerHead {
  std::size_t hidden{0};
  std::size_t dim{0};
  std::vector<float> q_weight, q_bias, k_weight, k_bias;  // [dim, hidden]
  std::vector<float> final_norm;                          // [hidden]
  float temperature{1.0F};
  float eps{1e-6F};

  /// Softmax over one question's options from final-layer residual rows
  /// (before the final norm).
  [[nodiscard]] std::vector<double> Probabilities(
      std::span<const float> decide,
      std::span<const std::span<const float>> options) const;
};

[[nodiscard]] bool LoadPointerHead(const core::GgufReader& reader,
                                   PointerHead* head, std::string* error);

struct ModelOptions {
  Limits limits;
  PackedOptions packed;
  /// Context of the unbatched reference executor (tests only).
  std::uint32_t baseline_context{16384};
};

class Model {
public:
  ~Model();
  [[nodiscard]] static std::unique_ptr<Model> Load(
      const std::filesystem::path& path, const ModelOptions& options,
      std::string* error);

  [[nodiscard]] const Encoder& encoder() const noexcept { return *encoder_; }
  [[nodiscard]] const Limits& limits() const noexcept { return limits_; }
  [[nodiscard]] const PointerHead& head() const noexcept { return head_; }
  [[nodiscard]] PackedStats stats() const noexcept { return packed_->stats(); }
  [[nodiscard]] std::size_t ReservedBytes() const noexcept {
    return packed_->ReservedBytes();
  }
  /// See PackedExecutor::SetSliceHook.
  void SetSliceHook(std::function<void(std::uint32_t)> hook,
                    std::uint32_t layers) {
    packed_->SetSliceHook(std::move(hook), layers);
  }

  /// Probabilities per question, in request order.
  [[nodiscard]] std::vector<std::vector<double>> Score(
      const Encoding& encoding);
  /// Probabilities per request, per question, from shared packed passes.
  [[nodiscard]] std::vector<std::vector<std::vector<double>>> ScoreBatch(
      std::span<const Encoding* const> batch);
  /// The unbatched reference: the state once on the Qwen executor, then each
  /// question row after restoring its recurrent state.
  [[nodiscard]] std::vector<std::vector<double>> ScoreBaseline(
      const Encoding& encoding);

private:
  Model() = default;
  std::shared_ptr<const core::GgufReader> reader_;
  std::unique_ptr<Encoder> encoder_;
  std::shared_ptr<const ::gufo::hip::QwenGpuModel> gpu_model_;
  std::unique_ptr<PackedExecutor> packed_;
  std::unique_ptr<::gufo::hip::QwenGpuExecutor> baseline_;
  PointerHead head_;
  Limits limits_;
  std::uint32_t baseline_context_{0};
};

}  // namespace gufo::models::kev

#endif  // GUFO_MODELS_KEV_MODEL_HPP_

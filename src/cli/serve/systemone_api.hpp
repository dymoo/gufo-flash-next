#ifndef GUFO_SERVER_SYSTEMONE_API_HPP_
#define GUFO_SERVER_SYSTEMONE_API_HPP_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/kev/encoding.hpp"
#include "src/models/kev/hip/packed.hpp"

namespace gufo::models::kev {
class Model;
}

namespace gufo::server {

struct HttpRequest;
struct HttpResponse;

/// TypeSafe System One (`POST /v1/systemone`) served by Kev.
class SystemOneService {
public:
  /// Probabilities per request, per question, for one batch.
  using Scorer = std::function<std::vector<std::vector<std::vector<double>>>(
      std::span<const models::kev::Encoding* const>)>;

  struct Options {
    std::string model_id{"kev-4b"};
    std::string release_date;
    std::size_t max_pending{256};
    /// Tokens the worker takes into one batch (the model splits it into
    /// passes).
    std::size_t max_batch_tokens{32768};
    /// Kev's share of wall time on the GPU while the text model is busy:
    /// after each slice of layers Kev pauses for slice * (1 - share) / share.
    double gpu_share{0.25};
  };

  SystemOneService(std::unique_ptr<models::kev::Model> model, Options options);
  /// Test seam: an encoder and a scorer instead of a GPU model.
  SystemOneService(const models::kev::Encoder* encoder,
                   models::kev::Limits limits, Scorer scorer, Options options);
  ~SystemOneService();

  [[nodiscard]] const std::string& model_id() const noexcept {
    return options_.model_id;
  }
  [[nodiscard]] HttpResponse Handle(const HttpRequest& request);
  /// TypeSafe model cards for GET /v1/models.
  [[nodiscard]] json::Value Cards() const;
  /// The `systemone` section of GET /v1/runtime.
  [[nodiscard]] json::Value RuntimeJson() const;
  /// `gufo_systemone_*` Prometheus series for /metrics.
  [[nodiscard]] std::string PrometheusText() const;
  /// Tells Kev when the text model sharing its GPU has work in flight.
  void SetTextBusy(std::function<bool()> busy);

private:
  struct Job;
  void Work();
  void Pace(std::uint32_t layer);

  std::unique_ptr<models::kev::Model> model_;
  const models::kev::Encoder* encoder_{nullptr};
  models::kev::Limits limits_;
  Scorer scorer_;
  Options options_;
  // One worker owns the GPU: everything queued when it frees up becomes one
  // batch.
  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<Job*> queue_;
  bool stopping_{false};
  /// The model's pass and bank counters after the latest batch.
  models::kev::PackedStats bank_;
  std::function<bool()> text_busy_;                    // guarded by mutex_
  std::chrono::steady_clock::time_point slice_start_;  // worker thread only
  std::atomic<std::uint64_t> paced_ms_{0};
  std::atomic<std::uint64_t> batches_{0};
  std::atomic<std::size_t> pending_{0};
  std::atomic<std::uint64_t> requests_{0};
  std::atomic<std::uint64_t> rejected_{0};
  std::atomic<std::uint64_t> tokens_{0};
  std::atomic<std::uint64_t> questions_{0};
  std::thread worker_;
};

/// The {"detail": ...} error TypeSafe clients read.
[[nodiscard]] HttpResponse SystemOneError(int status, std::string detail);

}  // namespace gufo::server

#endif  // GUFO_SERVER_SYSTEMONE_API_HPP_

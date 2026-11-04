#ifndef GUFO_SERVER_METRICS_REGISTRY_HPP_
#define GUFO_SERVER_METRICS_REGISTRY_HPP_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>

namespace gufo::server::metrics {

// Bounded label sets. Every counter this file defines is indexed by one of
// these small enums (never a request id or a free-form string), so the
// exposition stays a fixed, known-size document regardless of traffic.
enum class Endpoint : std::uint8_t {
  kCompletion,
  kChatCompletions,
  kResponses,
  kMessages,
  kOther,
};
inline constexpr std::size_t kEndpointCount = 5;
inline constexpr std::string_view EndpointName(Endpoint endpoint) noexcept {
  switch (endpoint) {
    case Endpoint::kCompletion:
      return "completion";
    case Endpoint::kChatCompletions:
      return "chat.completions";
    case Endpoint::kResponses:
      return "responses";
    case Endpoint::kMessages:
      return "messages";
    case Endpoint::kOther:
      return "other";
  }
  return "other";
}

enum class Outcome : std::uint8_t {
  kCompleted,
  kCancelled,
  kRejectedBusy,
  kRejectedFlex,
  kRejectedCapacity,
  kContextExceeded,
  kError,
};
inline constexpr std::size_t kOutcomeCount = 7;
inline constexpr std::string_view OutcomeName(Outcome outcome) noexcept {
  switch (outcome) {
    case Outcome::kCompleted:
      return "completed";
    case Outcome::kCancelled:
      return "cancelled";
    case Outcome::kRejectedBusy:
      return "rejected_busy";
    case Outcome::kRejectedFlex:
      return "rejected_flex";
    case Outcome::kRejectedCapacity:
      return "rejected_capacity";
    case Outcome::kContextExceeded:
      return "context_exceeded";
    case Outcome::kError:
      return "error";
  }
  return "error";
}

enum class Tier : std::uint8_t { kRam, kDisk };
inline constexpr std::size_t kTierCount = 2;
inline constexpr std::string_view TierName(Tier tier) noexcept {
  return tier == Tier::kRam ? "ram" : "disk";
}

/// Why a request found no reusable prefix (ContinuationLookup::miss_reason).
inline constexpr std::array<std::string_view, 5> kMissReasons = {
    "no_checkpoint", "prefix_changed", "input_changed", "disabled", "other"};
inline constexpr std::size_t MissReasonIndex(std::string_view reason) noexcept {
  for (std::size_t index = 0; index + 1 < kMissReasons.size(); ++index) {
    if (kMissReasons[index] == reason)
      return index;
  }
  return kMissReasons.size() - 1;
}

/// Fixed-bucket histogram. Observe() is O(1): a linear scan over at most a
/// couple dozen bounds plus three relaxed atomic increments. No label
/// support -- a caller needing labels keeps one Histogram per label value,
/// which is exactly what the bounded Tier/Endpoint sets above are for.
template<std::size_t N>
class Histogram {
public:
  explicit constexpr Histogram(std::array<double, N> bounds) noexcept
      : bounds_(bounds) {}

  void Observe(double value) noexcept {
    std::size_t index = 0;
    while (index < N && value > bounds_[index])
      ++index;
    buckets_[index].fetch_add(1, std::memory_order_relaxed);
    count_.fetch_add(1, std::memory_order_relaxed);
    if (value > 0.0) {
      // Summed in integer microunits so the running sum stays a plain
      // lock-free atomic<uint64_t> instead of a CAS loop over atomic<double>.
      sum_micro_.fetch_add(static_cast<std::uint64_t>(value * 1e6),
                           std::memory_order_relaxed);
    }
  }

  void Write(std::ostream& out, std::string_view name) const {
    std::uint64_t cumulative = 0;
    for (std::size_t index = 0; index < N; ++index) {
      cumulative += buckets_[index].load(std::memory_order_relaxed);
      out << name << "_bucket{le=\"" << bounds_[index] << "\"} " << cumulative
          << "\n";
    }
    cumulative += buckets_[N].load(std::memory_order_relaxed);
    out << name << "_bucket{le=\"+Inf\"} " << cumulative << "\n";
    out << name << "_sum "
        << (static_cast<double>(sum_micro_.load(std::memory_order_relaxed)) /
            1e6)
        << "\n";
    out << name << "_count " << count_.load(std::memory_order_relaxed) << "\n";
  }

private:
  std::array<double, N> bounds_;
  std::array<std::atomic<std::uint64_t>, N + 1> buckets_{};
  std::atomic<std::uint64_t> count_{0};
  std::atomic<std::uint64_t> sum_micro_{0};
};

/// All gufo_* counters and histograms. A fixed, known-at-compile-time set of
/// named members -- not a generic name-keyed registry -- because the set of
/// metrics never grows at runtime. Gauges live outside this struct: they are
/// cheap to read at scrape time straight from their owning subsystem (the
/// scheduler, the VMM pool, /sys), so nothing needs to push into an atomic
/// on every request.
struct Registry {
  Registry(const Registry&) = delete;
  Registry& operator=(const Registry&) = delete;

  static Registry& Get() {
    static Registry registry;
    return registry;
  }

  std::array<std::array<std::atomic<std::uint64_t>, kOutcomeCount>,
             kEndpointCount>
      requests_total{};

  std::atomic<std::uint64_t> prompt_tokens_cached_total{0};
  std::atomic<std::uint64_t> prompt_tokens_uncached_total{0};
  std::atomic<std::uint64_t> generated_tokens_total{0};
  std::atomic<std::uint64_t> prefill_chunks_total{0};

  /// Indexed by TextSpeculativeBackend-derived label at exposition time; the
  /// backend is fixed for the process lifetime, so one running counter pair
  /// plus the configured label covers "by source" without per-token
  /// bookkeeping of which drafter proposed a given token.
  std::atomic<std::uint64_t> draft_tokens_proposed_total{0};
  std::atomic<std::uint64_t> draft_tokens_accepted_total{0};

  /// [tier][0]=hit, [tier][1]=miss.
  std::array<std::array<std::atomic<std::uint64_t>, 2>, kTierCount>
      prefix_cache_lookups_total{};
  std::array<std::atomic<std::uint64_t>, kTierCount>
      prefix_cache_restored_bytes_total{};
  std::array<std::atomic<std::uint64_t>, kMissReasons.size()>
      prefix_cache_misses_total{};
  /// Gauge set by the cache tiers whenever pins change.
  std::array<std::atomic<std::uint64_t>, kTierCount>
      prefix_cache_pinned_bytes{};

  std::array<std::atomic<std::uint64_t>, kTierCount> session_pageouts_total{};
  std::array<std::atomic<std::uint64_t>, kTierCount>
      session_pageout_bytes_total{};
  std::array<std::atomic<std::uint64_t>, kTierCount> session_pageins_total{};
  std::array<std::atomic<std::uint64_t>, kTierCount>
      session_pagein_bytes_total{};
  std::array<std::atomic<std::uint64_t>, kTierCount> snapshots_saved_total{};
  std::array<std::atomic<std::uint64_t>, kTierCount> snapshot_bytes_total{};

  Histogram<13> time_to_first_token_seconds{
      {0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60}};
  Histogram<13> inter_token_latency_seconds{
      {0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10}};
  Histogram<13> queue_wait_seconds{
      {0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60}};
  Histogram<13> request_duration_seconds{
      {0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600}};
  Histogram<9> prefill_chunk_tokens{
      {16, 32, 64, 128, 256, 512, 1024, 2048, 4096}};
  Histogram<10> decode_batch_width{{1, 2, 3, 4, 6, 8, 12, 16, 24, 32}};
  Histogram<11> prefix_cache_restore_seconds{
      {0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1}};

private:
  Registry() = default;
};

/// Gauges and build-info labels, read fresh at every scrape from whatever
/// subsystem owns the number (scheduler, VMM pool, /sys). A missing optional
/// gauge is simply not emitted, so a CPU-only or non-AMD host still returns
/// a valid, smaller exposition instead of a fabricated zero.
struct GaugeSnapshot {
  std::uint64_t sessions_configured{0};
  std::uint64_t sessions_resident{0};
  std::uint64_t sessions_active{0};
  std::uint64_t sessions_queued{0};
  std::uint64_t sessions_paged_out{0};
  std::uint64_t kv_pool_bytes_used{0};
  std::uint64_t kv_pool_bytes_limit{0};
  std::uint64_t host_snapshot_bytes{0};
  std::uint64_t host_snapshot_bytes_limit{0};
  std::uint64_t disk_cache_bytes{0};
  std::uint64_t disk_cache_bytes_limit{0};
  std::optional<std::uint64_t> gpu_gtt_bytes_used;
  std::optional<std::uint64_t> gpu_gtt_bytes_total;
  std::optional<double> gpu_edge_temperature_celsius;
  std::optional<double> gpu_sclk_mhz;
  std::string revision{"development"};
  std::string model{"unknown"};
  std::string kv_cache{"unknown"};
  std::string moe_routing{"unknown"};
  std::string speculative{"off"};
};

inline void WriteCounter2D(
    std::ostream& out, std::string_view name, std::string_view help,
    std::string_view label_a, std::string_view label_b,
    const std::array<std::array<std::atomic<std::uint64_t>, kOutcomeCount>,
                     kEndpointCount>& values) {
  out << "# HELP " << name << " " << help << "\n# TYPE " << name
      << " counter\n";
  for (std::size_t endpoint = 0; endpoint < kEndpointCount; ++endpoint) {
    for (std::size_t outcome = 0; outcome < kOutcomeCount; ++outcome) {
      out << name << "{" << label_a << "=\""
          << EndpointName(static_cast<Endpoint>(endpoint)) << "\"," << label_b
          << "=\"" << OutcomeName(static_cast<Outcome>(outcome)) << "\"} "
          << values[endpoint][outcome].load(std::memory_order_relaxed) << "\n";
    }
  }
}

inline void WriteCounterByTier(
    std::ostream& out, std::string_view name, std::string_view help,
    const std::array<std::atomic<std::uint64_t>, kTierCount>& values) {
  out << "# HELP " << name << " " << help << "\n# TYPE " << name
      << " counter\n";
  for (std::size_t tier = 0; tier < kTierCount; ++tier) {
    out << name << "{tier=\"" << TierName(static_cast<Tier>(tier)) << "\"} "
        << values[tier].load(std::memory_order_relaxed) << "\n";
  }
}

inline void WriteGauge(std::ostream& out, std::string_view name,
                       std::string_view help, double value) {
  out << "# HELP " << name << " " << help << "\n# TYPE " << name << " gauge\n"
      << name << " " << value << "\n";
}

inline void WriteExposition(std::ostream& out, const Registry& registry,
                            const GaugeSnapshot& gauges) {
  WriteCounter2D(out, "gufo_requests_total",
                 "Total HTTP requests by endpoint and outcome", "endpoint",
                 "outcome", registry.requests_total);

  out << "# HELP gufo_prompt_tokens_total Total prompt tokens processed by "
         "cache state\n# TYPE gufo_prompt_tokens_total counter\n"
      << "gufo_prompt_tokens_total{state=\"cached\"} "
      << registry.prompt_tokens_cached_total.load(std::memory_order_relaxed)
      << "\ngufo_prompt_tokens_total{state=\"uncached\"} "
      << registry.prompt_tokens_uncached_total.load(std::memory_order_relaxed)
      << "\n";

  out << "# HELP gufo_generated_tokens_total Total tokens generated\n"
         "# TYPE gufo_generated_tokens_total counter\n"
         "gufo_generated_tokens_total "
      << registry.generated_tokens_total.load(std::memory_order_relaxed)
      << "\n";

  out << "# HELP gufo_prefill_chunks_total Total prefill chunks processed\n"
         "# TYPE gufo_prefill_chunks_total counter\n"
         "gufo_prefill_chunks_total "
      << registry.prefill_chunks_total.load(std::memory_order_relaxed) << "\n";

  out << "# HELP gufo_draft_tokens_proposed_total Speculative draft tokens "
         "proposed\n# TYPE gufo_draft_tokens_proposed_total counter\n"
      << "gufo_draft_tokens_proposed_total{source=\"" << gauges.speculative
      << "\"} "
      << registry.draft_tokens_proposed_total.load(std::memory_order_relaxed)
      << "\n";
  out << "# HELP gufo_draft_tokens_accepted_total Speculative draft tokens "
         "accepted\n# TYPE gufo_draft_tokens_accepted_total counter\n"
      << "gufo_draft_tokens_accepted_total{source=\"" << gauges.speculative
      << "\"} "
      << registry.draft_tokens_accepted_total.load(std::memory_order_relaxed)
      << "\n";

  out << "# HELP gufo_prefix_cache_lookups_total Continuation-cache lookups "
         "by tier and result\n# TYPE gufo_prefix_cache_lookups_total "
         "counter\n";
  for (std::size_t tier = 0; tier < kTierCount; ++tier) {
    out << "gufo_prefix_cache_lookups_total{tier=\""
        << TierName(static_cast<Tier>(tier)) << "\",result=\"hit\"} "
        << registry.prefix_cache_lookups_total[tier][0].load(
               std::memory_order_relaxed)
        << "\ngufo_prefix_cache_lookups_total{tier=\""
        << TierName(static_cast<Tier>(tier)) << "\",result=\"miss\"} "
        << registry.prefix_cache_lookups_total[tier][1].load(
               std::memory_order_relaxed)
        << "\n";
  }
  WriteCounterByTier(out, "gufo_prefix_cache_restored_bytes_total",
                     "Bytes restored from the continuation cache by tier",
                     registry.prefix_cache_restored_bytes_total);
  out << "# HELP gufo_prefix_cache_misses_total Requests without a reusable "
         "prefix by reason\n# TYPE gufo_prefix_cache_misses_total counter\n";
  for (std::size_t reason = 0; reason < kMissReasons.size(); ++reason) {
    out << "gufo_prefix_cache_misses_total{reason=\"" << kMissReasons[reason]
        << "\"} "
        << registry.prefix_cache_misses_total[reason].load(
               std::memory_order_relaxed)
        << "\n";
  }
  out << "# HELP gufo_prefix_cache_pinned_bytes Shared-prefix snapshot bytes "
         "exempt from eviction by tier\n# TYPE gufo_prefix_cache_pinned_bytes "
         "gauge\n";
  for (std::size_t tier = 0; tier < kTierCount; ++tier) {
    out << "gufo_prefix_cache_pinned_bytes{tier=\""
        << TierName(static_cast<Tier>(tier)) << "\"} "
        << registry.prefix_cache_pinned_bytes[tier].load(
               std::memory_order_relaxed)
        << "\n";
  }

  WriteCounterByTier(out, "gufo_session_pageouts_total",
                     "Sessions paged out of a tier",
                     registry.session_pageouts_total);
  WriteCounterByTier(out, "gufo_session_pageout_bytes_total",
                     "Bytes paged out of a tier",
                     registry.session_pageout_bytes_total);
  WriteCounterByTier(out, "gufo_session_pageins_total",
                     "Sessions paged into a tier",
                     registry.session_pageins_total);
  WriteCounterByTier(out, "gufo_session_pagein_bytes_total",
                     "Bytes paged into a tier",
                     registry.session_pagein_bytes_total);
  WriteCounterByTier(out, "gufo_snapshots_saved_total",
                     "Continuation snapshots saved by tier",
                     registry.snapshots_saved_total);
  WriteCounterByTier(out, "gufo_snapshot_bytes_total",
                     "Continuation snapshot bytes saved by tier",
                     registry.snapshot_bytes_total);

  out << "# HELP gufo_time_to_first_token_seconds Time from request "
         "admission to the first generated token\n# TYPE "
         "gufo_time_to_first_token_seconds histogram\n";
  registry.time_to_first_token_seconds.Write(
      out, "gufo_time_to_first_token_seconds");
  out << "# HELP gufo_inter_token_latency_seconds Mean time between "
         "generated tokens within a request\n# TYPE "
         "gufo_inter_token_latency_seconds histogram\n";
  registry.inter_token_latency_seconds.Write(
      out, "gufo_inter_token_latency_seconds");
  out << "# HELP gufo_queue_wait_seconds Time a request spent queued before "
         "admission\n# TYPE gufo_queue_wait_seconds histogram\n";
  registry.queue_wait_seconds.Write(out, "gufo_queue_wait_seconds");
  out << "# HELP gufo_request_duration_seconds Queue plus prefill plus "
         "decode wall time\n# TYPE gufo_request_duration_seconds "
         "histogram\n";
  registry.request_duration_seconds.Write(out, "gufo_request_duration_seconds");
  out << "# HELP gufo_prefill_chunk_tokens Tokens processed per prefill "
         "chunk\n# TYPE gufo_prefill_chunk_tokens histogram\n";
  registry.prefill_chunk_tokens.Write(out, "gufo_prefill_chunk_tokens");
  out << "# HELP gufo_decode_batch_width Sessions batched together per "
         "decode step\n# TYPE gufo_decode_batch_width histogram\n";
  registry.decode_batch_width.Write(out, "gufo_decode_batch_width");
  out << "# HELP gufo_prefix_cache_restore_seconds Latency of restoring a "
         "continuation-cache hit\n# TYPE gufo_prefix_cache_restore_seconds "
         "histogram\n";
  registry.prefix_cache_restore_seconds.Write(
      out, "gufo_prefix_cache_restore_seconds");

  WriteGauge(out, "gufo_sessions_configured",
             "Configured concurrent session slots",
             static_cast<double>(gauges.sessions_configured));
  WriteGauge(out, "gufo_sessions_resident",
             "Sessions currently holding a runner slot",
             static_cast<double>(gauges.sessions_resident));
  WriteGauge(out, "gufo_sessions_active", "Sessions currently decoding",
             static_cast<double>(gauges.sessions_active));
  WriteGauge(out, "gufo_sessions_queued", "Requests waiting for admission",
             static_cast<double>(gauges.sessions_queued));
  WriteGauge(out, "gufo_sessions_paged_out",
             "Sessions currently resident only on the disk cache",
             static_cast<double>(gauges.sessions_paged_out));
  WriteGauge(out, "gufo_kv_pool_bytes_used", "Device KV pool bytes in use",
             static_cast<double>(gauges.kv_pool_bytes_used));
  WriteGauge(out, "gufo_kv_pool_bytes_limit", "Device KV pool byte limit",
             static_cast<double>(gauges.kv_pool_bytes_limit));
  WriteGauge(out, "gufo_host_snapshot_bytes",
             "Host RAM bytes retained for continuation snapshots",
             static_cast<double>(gauges.host_snapshot_bytes));
  WriteGauge(out, "gufo_host_snapshot_bytes_limit", "Host RAM snapshot budget",
             static_cast<double>(gauges.host_snapshot_bytes_limit));
  WriteGauge(out, "gufo_disk_cache_bytes", "Disk cache bytes retained",
             static_cast<double>(gauges.disk_cache_bytes));
  WriteGauge(out, "gufo_disk_cache_bytes_limit", "Disk cache byte capacity",
             static_cast<double>(gauges.disk_cache_bytes_limit));
  if (gauges.gpu_gtt_bytes_used) {
    WriteGauge(out, "gufo_gpu_gtt_bytes_used", "GPU GTT memory in use",
               static_cast<double>(*gauges.gpu_gtt_bytes_used));
  }
  if (gauges.gpu_gtt_bytes_total) {
    WriteGauge(out, "gufo_gpu_gtt_bytes_total", "GPU GTT memory total",
               static_cast<double>(*gauges.gpu_gtt_bytes_total));
  }
  if (gauges.gpu_edge_temperature_celsius) {
    WriteGauge(out, "gufo_gpu_edge_temperature_celsius", "GPU edge temperature",
               *gauges.gpu_edge_temperature_celsius);
  }
  if (gauges.gpu_sclk_mhz) {
    WriteGauge(out, "gufo_gpu_sclk_mhz", "GPU current shader clock",
               *gauges.gpu_sclk_mhz);
  }

  out << "# HELP gufo_build_info Static build and configuration labels\n"
         "# TYPE gufo_build_info gauge\n"
         "gufo_build_info{revision=\""
      << gauges.revision << "\",model=\"" << gauges.model << "\",kv_cache=\""
      << gauges.kv_cache << "\",moe_routing=\"" << gauges.moe_routing
      << "\",speculative=\"" << gauges.speculative << "\"} 1\n";
}

}  // namespace gufo::server::metrics

#endif  // GUFO_SERVER_METRICS_REGISTRY_HPP_

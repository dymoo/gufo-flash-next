#include "src/cli/serve/metrics_registry.hpp"

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

namespace {

using gufo::server::metrics::Endpoint;
using gufo::server::metrics::GaugeSnapshot;
using gufo::server::metrics::Histogram;
using gufo::server::metrics::MissReasonIndex;
using gufo::server::metrics::Outcome;
using gufo::server::metrics::Registry;
using gufo::server::metrics::Tier;
using gufo::server::metrics::WriteExposition;

void TestHistogramBucketsAndSum() {
  Histogram<3> histogram({1.0, 2.0, 4.0});
  histogram.Observe(0.5);    // bucket 0 (<=1)
  histogram.Observe(1.5);    // bucket 1 (<=2)
  histogram.Observe(4.0);    // bucket 2 (<=4)
  histogram.Observe(100.0);  // overflow (+Inf)

  std::ostringstream out;
  histogram.Write(out, "test_metric");
  const std::string text = out.str();
  assert(text.find(R"(test_metric_bucket{le="1"} 1)") != std::string::npos);
  assert(text.find(R"(test_metric_bucket{le="2"} 2)") != std::string::npos);
  assert(text.find(R"(test_metric_bucket{le="4"} 3)") != std::string::npos);
  assert(text.find(R"(test_metric_bucket{le="+Inf"} 4)") != std::string::npos);
  assert(text.find("test_metric_count 4") != std::string::npos);
  assert(text.find("test_metric_sum 106") != std::string::npos);
}

void TestExpositionIsWellFormed() {
  // The registry is a process-wide singleton; read the deltas this test
  // itself produces rather than assuming a pristine zero state.
  auto& registry = Registry::Get();
  const auto before =
      registry
          .requests_total[static_cast<std::size_t>(Endpoint::kChatCompletions)]
                         [static_cast<std::size_t>(Outcome::kCompleted)]
          .load();
  registry
      .requests_total[static_cast<std::size_t>(Endpoint::kChatCompletions)]
                     [static_cast<std::size_t>(Outcome::kCompleted)]
      .fetch_add(1);
  registry.prefix_cache_lookups_total[static_cast<std::size_t>(Tier::kDisk)][0]
      .fetch_add(3);
  registry.generated_tokens_total.fetch_add(42);
  const auto changed =
      registry.prefix_cache_misses_total[MissReasonIndex("prefix_changed")]
          .load();
  registry.prefix_cache_misses_total[MissReasonIndex("prefix_changed")]
      .fetch_add(1);
  assert(MissReasonIndex("unexpected") == MissReasonIndex("other"));
  registry.prefix_cache_pinned_bytes[static_cast<std::size_t>(Tier::kDisk)]
      .store(512);

  GaugeSnapshot gauges;
  gauges.sessions_configured = 4;
  gauges.kv_pool_bytes_used = 1024;
  gauges.kv_pool_bytes_limit = 4096;
  gauges.revision = "test-rev";
  gauges.model = "test-model";

  std::ostringstream out;
  WriteExposition(out, registry, gauges);
  const std::string text = out.str();

  // Every metric declares HELP and TYPE before its samples (Prometheus text
  // exposition 0.0.4), and the two sample lines this test just wrote appear.
  assert(text.find("# HELP gufo_requests_total") != std::string::npos);
  assert(text.find("# TYPE gufo_requests_total counter") != std::string::npos);
  assert(
      text.find(
          R"(gufo_requests_total{endpoint="chat.completions",outcome="completed"} )" +
          std::to_string(before + 1)) != std::string::npos);
  assert(
      text.find(
          R"(gufo_prefix_cache_lookups_total{tier="disk",result="hit"} 3)") !=
      std::string::npos);
  assert(
      text.find(R"(gufo_prefix_cache_misses_total{reason="prefix_changed"} )" +
                std::to_string(changed + 1)) != std::string::npos);
  assert(text.find(R"(gufo_prefix_cache_pinned_bytes{tier="disk"} 512)") !=
         std::string::npos);
  assert(text.find("gufo_generated_tokens_total 42") != std::string::npos);
  assert(text.find("gufo_kv_pool_bytes_used 1024") != std::string::npos);
  assert(text.find("gufo_kv_pool_bytes_limit 4096") != std::string::npos);
  assert(
      text.find(R"(gufo_build_info{revision="test-rev",model="test-model")") !=
      std::string::npos);
  // A gauge without a reading (no GPU sysfs on this host) is simply absent,
  // never a fabricated zero.
  assert(text.find("gufo_gpu_gtt_bytes_used") == std::string::npos);
}

}  // namespace

int main() {
  TestHistogramBucketsAndSum();
  TestExpositionIsWellFormed();
  std::cout << "Metrics registry checks passed.\n";
}

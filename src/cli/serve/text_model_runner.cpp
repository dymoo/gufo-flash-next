#include "src/cli/serve/text_model_runner.hpp"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "src/cli/serve/continuation_disk_store.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/metrics_registry.hpp"

namespace gufo::server {

namespace {
std::atomic<std::size_t> explicit_host_budget{0};

/// Host RAM available to this process: MemAvailable bounded by every
/// enclosing cgroup's limit.
std::size_t HostAvailableBytes() {
  const long pages = sysconf(_SC_AVPHYS_PAGES);
  const long page_size = sysconf(_SC_PAGESIZE);
  if (pages <= 0 || page_size <= 0)
    return 0;
  std::uint64_t available = std::uint64_t(pages) * page_size;
  std::ifstream meminfo("/proc/meminfo");
  for (std::string line; std::getline(meminfo, line);) {
    if (line.starts_with("MemAvailable:")) {
      std::istringstream fields(line.substr(13));
      std::uint64_t kib = 0;
      if (fields >> kib)
        available = kib * 1024;
      break;
    }
  }
  // A cgroup limit can be much smaller than the host's available memory.
  // Walk parents too: a child may say "max" beneath a limited ancestor.
  std::ifstream membership("/proc/self/cgroup");
  for (std::string line; std::getline(membership, line);) {
    if (!line.starts_with("0::/"))
      continue;
    const auto relative = std::filesystem::path(line.substr(4));
    if (std::ranges::any_of(relative,
                            [](const auto& part) { return part == ".."; }))
      continue;
    const std::filesystem::path root("/sys/fs/cgroup");
    for (auto path = root / relative;; path = path.parent_path()) {
      std::ifstream limit_file(path / "memory.max");
      std::ifstream used_file(path / "memory.current");
      std::uint64_t limit = 0, used = 0;
      if ((limit_file >> limit) && (used_file >> used))
        available = std::min(available, limit > used ? limit - used : 0);
      if (path == root)
        break;
    }
    break;
  }
  return static_cast<std::size_t>(std::min<std::uint64_t>(
      available, std::numeric_limits<std::size_t>::max()));
}
}  // namespace

void SetHostMemoryBudget(std::size_t bytes) noexcept {
  explicit_host_budget = bytes;
}

bool HostMemoryBudgetExplicit() noexcept {
  return explicit_host_budget != 0;
}

std::size_t HostMemoryBudgetBytes() {
  if (explicit_host_budget != 0)
    return explicit_host_budget;
  // Keep 8 GiB (a quarter in small cgroups) for everything else.
  const auto available = HostAvailableBytes();
  return available - std::min(std::size_t{8} << 30, available / 4);
}

/// Ordinary host allocations use the host budget, not HIP's device capacity.
std::size_t HostSnapshotBudgetBytes() {
  return HostMemoryBudgetBytes() / 2;
}

namespace {

struct ValidatedRunner {
  std::shared_ptr<TextModelRunner> runner;
  TextRunnerDescriptor descriptor;
  TextRunnerResourceClaim resources;
  std::vector<TextExecutionPlan> plans;
};

std::optional<std::size_t> PerStateReservationBytes(
    const TextRunnerResourceClaim& resources) {
  if (!resources.per_request_state_bytes.has_value() ||
      !resources.temporary_scratch_bytes.has_value()) {
    return std::nullopt;
  }
  if (*resources.per_request_state_bytes >
      std::numeric_limits<std::size_t>::max() -
          *resources.temporary_scratch_bytes) {
    throw std::invalid_argument("text runner resource claim overflows");
  }
  return *resources.per_request_state_bytes +
         *resources.temporary_scratch_bytes;
}

ValidatedRunner ValidateRunner(std::shared_ptr<TextModelRunner> runner,
                               std::size_t state_count) {
  if (runner == nullptr) {
    throw std::invalid_argument("text model runner must not be null");
  }
  if (state_count == 0) {
    throw std::invalid_argument("text runner state count must be at least one");
  }

  auto descriptor = runner->Descriptor();
  if (descriptor.model_id.empty()) {
    throw std::invalid_argument("text runner model ID must not be empty");
  }
  if (descriptor.state_abi.empty()) {
    throw std::invalid_argument("text runner state ABI must not be empty");
  }
  if (descriptor.max_context == 0) {
    throw std::invalid_argument(
        "text runner maximum context must be at least one token");
  }
  if (descriptor.capabilities.fork && !descriptor.capabilities.snapshot) {
    throw std::invalid_argument(
        "text runner fork capability requires snapshot support");
  }
  if (descriptor.persistence.has_value()) {
    if (!descriptor.capabilities.snapshot || !descriptor.capabilities.fork) {
      throw std::invalid_argument(
          "text runner persistence requires snapshot and fork support");
    }
    if (descriptor.persistence->compatibility_identity.empty()) {
      throw std::invalid_argument(
          "text runner persistence identity must not be empty");
    }
    if (descriptor.persistence->payload_version == 0) {
      throw std::invalid_argument(
          "text runner persistence payload version must be nonzero");
    }
  }

  auto resources = runner->ResourceClaim();
  const auto per_state_reservation = PerStateReservationBytes(resources);
  if (resources.state_capacity_bytes.has_value() &&
      per_state_reservation.has_value()) {
    const std::size_t per_request = *per_state_reservation;
    const std::size_t capacity = *resources.state_capacity_bytes;
    if (per_request != 0 && state_count > capacity / per_request) {
      throw std::invalid_argument(
          "text runner request-state claim exceeds state capacity");
    }
  }

  auto plans = runner->SupportedPlans();
  if (plans.empty()) {
    throw std::invalid_argument(
        "text runner must expose at least one execution plan");
  }
  bool supports_serial_single = false;
  for (std::size_t index = 0; index < plans.size(); ++index) {
    const auto& plan = plans[index];
    if (plan.physical_width == 0) {
      throw std::invalid_argument(
          "text runner execution plan width must be at least one");
    }
    if ((plan.kind == TextExecutionPlanKind::kSerial &&
         plan.physical_width != 1) ||
        (plan.kind == TextExecutionPlanKind::kBatched &&
         plan.physical_width < 2)) {
      throw std::invalid_argument(
          "text runner execution plan kind and width are inconsistent");
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (plans[previous] == plan) {
        throw std::invalid_argument(
            "text runner execution plans must be unique");
      }
    }
    supports_serial_single = supports_serial_single ||
                             (plan.kind == TextExecutionPlanKind::kSerial &&
                              plan.physical_width == 1);
  }
  if (!supports_serial_single) {
    throw std::invalid_argument(
        "text runner must support the serial single-request plan");
  }

  return {
      .runner = std::move(runner),
      .descriptor = std::move(descriptor),
      .resources = resources,
      .plans = std::move(plans),
  };
}

void ReconcileStateBytes(const TextRunnerResourceClaim& resources,
                         const TextRunnerState& state) {
  const auto measured = state.MeasuredResources();
  if (measured.per_request_state_bytes.has_value() &&
      resources.per_request_state_bytes.has_value() &&
      *measured.per_request_state_bytes > *resources.per_request_state_bytes) {
    throw std::runtime_error(
        "text runner measured state exceeds its resource claim");
  }
  if (measured.temporary_scratch_bytes.has_value() &&
      resources.temporary_scratch_bytes.has_value() &&
      *measured.temporary_scratch_bytes > *resources.temporary_scratch_bytes) {
    throw std::runtime_error(
        "text runner measured scratch exceeds its resource claim");
  }
}

std::string_view SnapshotEventActionName(SnapshotEventAction action) noexcept {
  switch (action) {
    case SnapshotEventAction::kRemoved:
      return "removed";
    case SnapshotEventAction::kSkipped:
      return "skipped";
  }
  return "unknown";
}

std::string_view SnapshotEventReasonName(SnapshotEventReason reason) noexcept {
  switch (reason) {
    case SnapshotEventReason::kByteCapacity:
      return "byte_capacity";
    case SnapshotEventReason::kEntryCapacity:
      return "entry_capacity";
    case SnapshotEventReason::kExactReplacement:
      return "exact_replacement";
    case SnapshotEventReason::kCaptureFailure:
      return "capture_failure";
    case SnapshotEventReason::kReservationMismatch:
      return "reservation_mismatch";
  }
  return "unknown";
}

void EmitSnapshotEvent(const SnapshotEvent& event) noexcept {
  if (event.action == SnapshotEventAction::kRemoved) {
    auto& registry = metrics::Registry::Get();
    registry
        .session_pageouts_total[static_cast<std::size_t>(metrics::Tier::kRam)]
        .fetch_add(1, std::memory_order_relaxed);
    registry
        .session_pageout_bytes_total[static_cast<std::size_t>(
            metrics::Tier::kRam)]
        .fetch_add(event.snapshot_bytes, std::memory_order_relaxed);
  }
  // Replacing or evicting a retained prefix is routine; request summaries
  // already report whether reuse succeeded. Failed captures need attention.
  if (event.action == SnapshotEventAction::kRemoved)
    return;
  try {
    std::ostringstream line;
    line << "event=snapshot action=" << SnapshotEventActionName(event.action)
         << " reason=" << SnapshotEventReasonName(event.reason)
         << " bytes=" << event.snapshot_bytes << " tokens=" << event.token_count
         << " retained_bytes=" << event.retained_snapshot_bytes
         << " reserved_bytes=" << event.reserved_snapshot_bytes
         << " capacity_bytes=" << event.capacity_bytes;
    Logger::Warn("cache", line.str());
  } catch (...) {
    // Cache logging must not affect request execution.
  }
}

std::string_view DiskEventActionName(
    ContinuationDiskEventAction action) noexcept {
  switch (action) {
    case ContinuationDiskEventAction::kStored:
      return "stored";
    case ContinuationDiskEventAction::kRestored:
      return "restored";
    case ContinuationDiskEventAction::kMiss:
      return "miss";
    case ContinuationDiskEventAction::kRemoved:
      return "removed";
    case ContinuationDiskEventAction::kSkipped:
      return "skipped";
  }
  return "unknown";
}

std::string_view DiskEventReasonName(
    ContinuationDiskEventReason reason) noexcept {
  switch (reason) {
    case ContinuationDiskEventReason::kSaved:
      return "saved";
    case ContinuationDiskEventReason::kHit:
      return "hit";
    case ContinuationDiskEventReason::kNotFound:
      return "not_found";
    case ContinuationDiskEventReason::kUnsupported:
      return "unsupported";
    case ContinuationDiskEventReason::kByteCapacity:
      return "byte_capacity";
    case ContinuationDiskEventReason::kStagingCapacity:
      return "staging_capacity";
    case ContinuationDiskEventReason::kLru:
      return "lru";
    case ContinuationDiskEventReason::kExactReplacement:
      return "exact_replacement";
    case ContinuationDiskEventReason::kCorrupt:
      return "corrupt";
    case ContinuationDiskEventReason::kChecksumMismatch:
      return "checksum_mismatch";
    case ContinuationDiskEventReason::kUnsafeFile:
      return "unsafe_file";
    case ContinuationDiskEventReason::kIoFailure:
      return "io_failure";
    case ContinuationDiskEventReason::kSerializationFailure:
      return "serialization_failure";
    case ContinuationDiskEventReason::kRestoreFailure:
      return "restore_failure";
    case ContinuationDiskEventReason::kBusy:
      return "busy";
  }
  return "unknown";
}

/// FNV-1a over identity bytes and tokens: keys for pin bookkeeping only;
/// every restore still verifies exact tokens.
std::uint64_t HashTokens(std::span<const std::uint8_t> identity,
                         std::span<const TextRunnerToken> tokens) {
  std::uint64_t hash = 14695981039346656037ULL;
  const auto mix = [&hash](std::uint8_t byte) {
    hash = (hash ^ byte) * 1099511628211ULL;
  };
  for (const auto byte : identity)
    mix(byte);
  for (const auto token : tokens)
    for (unsigned shift = 0; shift < 32; shift += 8)
      mix(static_cast<std::uint8_t>(token >> shift));
  return hash;
}

void EmitDiskEvent(const ContinuationDiskEvent& event) noexcept {
  {
    auto& registry = metrics::Registry::Get();
    constexpr auto kDisk = static_cast<std::size_t>(metrics::Tier::kDisk);
    if (event.action == ContinuationDiskEventAction::kStored) {
      registry.snapshots_saved_total[kDisk].fetch_add(
          1, std::memory_order_relaxed);
      registry.snapshot_bytes_total[kDisk].fetch_add(event.payload_bytes,
                                                     std::memory_order_relaxed);
    } else if (event.action == ContinuationDiskEventAction::kRestored) {
      registry.session_pageins_total[kDisk].fetch_add(
          1, std::memory_order_relaxed);
      registry.session_pagein_bytes_total[kDisk].fetch_add(
          event.payload_bytes, std::memory_order_relaxed);
    } else if (event.action == ContinuationDiskEventAction::kRemoved) {
      registry.session_pageouts_total[kDisk].fetch_add(
          1, std::memory_order_relaxed);
      registry.session_pageout_bytes_total[kDisk].fetch_add(
          event.file_bytes, std::memory_order_relaxed);
    }
  }
  switch (event.reason) {
    case ContinuationDiskEventReason::kHit:
    case ContinuationDiskEventReason::kNotFound:
    case ContinuationDiskEventReason::kLru:
    case ContinuationDiskEventReason::kExactReplacement:
    case ContinuationDiskEventReason::kBusy:
      return;
    case ContinuationDiskEventReason::kByteCapacity:
      if (event.action == ContinuationDiskEventAction::kRemoved)
        return;
      break;
    default:
      break;
  }
  try {
    std::ostringstream line;
    line << "event=disk_cache action=" << DiskEventActionName(event.action)
         << " reason=" << DiskEventReasonName(event.reason)
         << " file_bytes=" << event.file_bytes
         << " payload_bytes=" << event.payload_bytes
         << " tokens=" << event.token_count
         << " retained_bytes=" << event.retained_bytes
         << " capacity_bytes=" << event.capacity_bytes
         << " staging_capacity_bytes=" << event.staging_capacity_bytes
         << " staging_used_bytes=" << event.staging_used_bytes;
    if (event.reason == ContinuationDiskEventReason::kSaved) {
      line << " write_ms=" << event.elapsed_ms;
      Logger::Info("cache", line.str());
    } else {
      Logger::Warn("cache", line.str());
    }
  } catch (...) {
    // Cache logging must not affect request execution.
  }
}

ContinuationCache::SnapshotSupport MakeSnapshotSupport(
    ValidatedRunner* validated) {
  if (validated == nullptr || !validated->descriptor.capabilities.snapshot ||
      !validated->descriptor.capabilities.fork) {
    return {};
  }
  return {
      .restore =
          [validated](ContinuationState& state,
                      const ContinuationSnapshot& snapshot) {
            auto* text_snapshot =
                dynamic_cast<const TextRunnerSnapshot*>(&snapshot);
            if (text_snapshot == nullptr) {
              throw std::invalid_argument(
                  "continuation snapshot is not a text runner snapshot");
            }
            auto& text_state = dynamic_cast<TextRunnerState&>(state);
            validated->runner->RestoreOrFork(text_state, *text_snapshot);
            ReconcileStateBytes(validated->resources, text_state);
          },
      .capacity_bytes =
          [validated] {
            const auto resources = validated->runner->ResourceClaim();
            return resources.retained_snapshot_capacity_bytes.value_or(0);
          },
      .on_event = EmitSnapshotEvent,
  };
}

}  // namespace

void TextModelRunner::AdvanceBatch(
    std::span<const TextRunnerAdvance> advances) const {
  for (const auto& advance : advances) {
    try {
      Advance(advance.state.get(), advance.token);
    } catch (...) {
      if (!advance.failure)
        throw;
      *advance.failure = std::current_exception();
    }
  }
}

TextDecodeStep TextModelRunner::DecodeStep(
    TextRunnerState& state, std::size_t max_tokens,
    sampling::SamplerState& sampler) const {
  if (max_tokens == 0) {
    throw std::invalid_argument(
        "text runner decode step budget must be at least one token");
  }
  auto selection = SelectNext(state, sampler);
  if (selection.stop) {
    return {
        .selections = {},
        .draft_tokens = 0,
        .draft_accepted_tokens = 0,
        .stop = true,
    };
  }
  const TextRunnerToken token = selection.token;
  Advance(state, token);
  return {
      .selections = {std::move(selection)},
  };
}

std::vector<TextDecodeStep> TextModelRunner::DecodeBatch(
    std::span<const TextRunnerDecode> decodes) const {
  std::vector<TextDecodeStep> steps;
  steps.reserve(decodes.size());
  for (const auto& decode : decodes) {
    try {
      steps.push_back(DecodeStep(decode.state.get(), decode.max_tokens,
                                 decode.sampler.get()));
    } catch (...) {
      TextDecodeStep failed;
      failed.failure = std::current_exception();
      steps.push_back(std::move(failed));
    }
  }
  return steps;
}

namespace {

void PrefillEach(const TextModelRunner& runner,
                 std::span<TextRunnerPrefill> prefills) {
  for (auto& prefill : prefills) {
    try {
      prefill.step = runner.Prefill(prefill.state.get(), prefill.prompt,
                                    prefill.offset, prefill.max_input_tokens);
    } catch (...) {
      prefill.failure = std::current_exception();
    }
  }
}

}  // namespace

std::vector<TextDecodeStep> TextModelRunner::DecodePrefillBatch(
    std::span<const TextRunnerDecode> decodes,
    std::span<TextRunnerPrefill> prefills) const {
  auto steps =
      decodes.empty() ? std::vector<TextDecodeStep>{} : DecodeBatch(decodes);
  PrefillEach(*this, prefills);
  return steps;
}

void TextModelRunner::AdvancePrefillBatch(
    std::span<const TextRunnerAdvance> advances,
    std::span<TextRunnerPrefill> prefills) const {
  if (!advances.empty())
    AdvanceBatch(advances);
  PrefillEach(*this, prefills);
}

std::unique_ptr<TextRunnerSnapshot> TextModelRunner::Snapshot(
    const TextRunnerState&) const {
  throw std::logic_error("text runner does not support snapshots");
}

std::size_t TextModelRunner::SnapshotPayloadBytes(
    const TextRunnerState&) const {
  throw std::logic_error("text runner does not support snapshot sizing");
}

void TextModelRunner::RestoreOrFork(TextRunnerState&,
                                    const TextRunnerSnapshot&) const {
  throw std::logic_error("text runner does not support snapshot restore/fork");
}

std::size_t TextModelRunner::PersistentSnapshotPayloadBytes(
    const TextRunnerSnapshot&) const {
  throw std::logic_error("text runner does not support persistent snapshots");
}

std::size_t TextModelRunner::SerializePersistentSnapshot(
    const TextRunnerSnapshot&, std::span<std::uint8_t>) const {
  throw std::logic_error("text runner does not support persistent snapshots");
}

void TextModelRunner::RestorePersistentSnapshot(
    TextRunnerState&, std::span<const std::uint8_t>) const {
  throw std::logic_error("text runner does not support persistent snapshots");
}

struct TextRunnerPool::Impl {
  Impl(std::shared_ptr<TextModelRunner> model_runner, std::size_t state_count,
       std::optional<TextRunnerDiskCacheOptions> disk_cache_options)
      : validated(ValidateRunner(std::move(model_runner), state_count)),
        cache(
            state_count,
            [this] {
              auto state = validated.runner->CreateState();
              if (state == nullptr) {
                throw std::runtime_error(
                    "text runner state factory returned null");
              }
              ReconcileStateBytes(validated.resources, *state);
              return state;
            },
            MakeSnapshotSupport(&validated)) {
    if (disk_cache_options.has_value()) {
      if (!validated.descriptor.persistence.has_value()) {
        throw std::invalid_argument(
            "text runner does not support persistent snapshots");
      }
      disk_store = std::make_shared<ContinuationDiskStore>(
          ContinuationDiskStoreOptions{
              .directory = std::move(disk_cache_options->directory),
              .capacity_bytes = disk_cache_options->capacity_bytes,
              .staging_capacity_bytes =
                  disk_cache_options->staging_capacity_bytes,
              .pin_capacity_bytes = disk_cache_options->pin_capacity_bytes,
          },
          EmitDiskEvent);
      Logger::Info(
          "cache",
          "event=disk_cache_configured capacity_bytes=" +
              std::to_string(disk_store->capacity_bytes()) +
              " staging_capacity_bytes=" +
              std::to_string(disk_store->staging_capacity_bytes()) +
              " pinned_bytes=" + std::to_string(disk_store->pinned_bytes()));
      shared_prefix_min_tokens = disk_cache_options->shared_prefix_min_tokens;
      shared_prefix_max_boundaries =
          disk_cache_options->shared_prefix_max_boundaries;
      pin_capacity_bytes = disk_cache_options->pin_capacity_bytes;
      pin_min_conversations = disk_cache_options->pin_min_conversations;
      PublishPinnedBytes();
    }
  }

  void PublishPinnedBytes() const noexcept {
    auto& pinned = metrics::Registry::Get().prefix_cache_pinned_bytes;
    pinned[static_cast<std::size_t>(metrics::Tier::kRam)].store(
        cache.pinned_snapshot_bytes(), std::memory_order_relaxed);
    if (disk_store)
      pinned[static_cast<std::size_t>(metrics::Tier::kDisk)].store(
          disk_store->pinned_bytes(), std::memory_order_relaxed);
  }

  /// Counts distinct conversations (by the tokens after the prefix) that
  /// start with prompt[0, prefix); from pin_min_conversations on, pins the
  /// prefix in each tier holding it. Called under admission_mutex.
  void ObserveSharedPrefix(std::span<const TextRunnerToken> prompt,
                           std::size_t prefix,
                           std::span<const std::uint8_t> identity) {
    if (disk_store == nullptr || pin_capacity_bytes == 0 ||
        pin_min_conversations == 0)
      return;
    // ponytail: forget progress wholesale when prompts never repeat.
    if (prefix_conversations.size() >= 1024)
      prefix_conversations.clear();
    const auto head = prompt.first(prefix);
    auto& seen = prefix_conversations[HashTokens(identity, head)];
    const auto tail = HashTokens(
        {}, prompt.subspan(prefix,
                           std::min<std::size_t>(64, prompt.size() - prefix)));
    if (seen.size() < pin_min_conversations &&
        std::ranges::find(seen, tail) == seen.end())
      seen.push_back(tail);
    if (seen.size() < pin_min_conversations)
      return;
    // Idempotent; retried per observation until each tier holds the prefix.
    (void)cache.Pin(
        head, identity,
        std::min(pin_capacity_bytes, cache.snapshot_capacity_bytes() / 2));
    (void)disk_store->Pin(*validated.runner, head, identity);
    PublishPinnedBytes();
  }

  ValidatedRunner validated;
  ContinuationCache cache;
  // Publish a live checkpoint before another admission can fall back to the
  // older prompt snapshot. Decode/prefill work never holds this mutex.
  std::timed_mutex admission_mutex;
  std::shared_ptr<ContinuationDiskStore> disk_store;
  std::size_t shared_prefix_min_tokens{
      TextRunnerDiskCacheOptions{}.shared_prefix_min_tokens};
  std::size_t shared_prefix_max_boundaries{0};
  std::size_t pin_capacity_bytes{0};
  std::size_t pin_min_conversations{0};
  /// Prefix hash -> hashes of distinct conversations seen after it.
  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>>
      prefix_conversations;
};

void TextModelRunner::StreamPersistentSnapshot(
    const TextRunnerSnapshot& snapshot, const SnapshotSink& sink) const {
  std::vector<std::uint8_t> payload(PersistentSnapshotPayloadBytes(snapshot));
  if (SerializePersistentSnapshot(snapshot, payload) != payload.size()) {
    throw std::runtime_error("persistent snapshot byte count mismatch");
  }
  sink(payload);
}

namespace {

/// The one place a generated token becomes part of a request: sampler
/// history and, for constrained requests, the grammar state.
void CommitToken(sampling::SamplerState& sampler, TextRunnerToken token) {
  sampler.Accept(token);
  if (const auto& constraint = sampler.config().constraint)
    constraint->Accept(token);
}

}  // namespace

struct TextRunnerPool::Request::Impl {
  Impl(std::shared_ptr<TextModelRunner> model_runner,
       std::shared_ptr<ContinuationDiskStore> persistent_store,
       ContinuationCache::Lease state_lease,
       std::vector<TextRunnerToken> prompt_tokens,
       std::vector<std::size_t> shared_prefix_boundaries,
       const sampling::SamplingConfig& sampling_config,
       std::shared_ptr<const TextPromptContext> prompt_context,
       std::size_t cache_prefix_tokens, std::size_t shared_prompt_prefix)
      : runner(std::move(model_runner)),
        disk_store(std::move(persistent_store)),
        lease(std::move(state_lease)),
        prompt(std::move(prompt_tokens)),
        boundaries(std::move(shared_prefix_boundaries)),
        shared_prefix(shared_prompt_prefix),
        prefill_offset(lease.cached_tokens()),
        decode_ready(prefill_offset == prompt.size()),
        sampler(sampling_config, prompt),
        context(std::move(prompt_context)) {
    const auto count =
        cache_prefix_tokens == 0 ? prompt.size() : cache_prefix_tokens;
    snapshot_tokens.assign(prompt.begin(), prompt.begin() + count);
    // A live continuation can already be beyond the desired snapshot point.
    // Keep its existing immutable checkpoint; never snapshot mismatched state.
    prompt_snapshot_attempted = prefill_offset > count;
    auto& state = dynamic_cast<TextRunnerState&>(lease.state());
    if (lease.cache_hit()) {
      runner->PreparePrefixReuse(
          state,
          std::span<const TextRunnerToken>(prompt).first(prefill_offset));
    }
  }

  ~Impl() {
    // The worker borrows this lease's state; join before any field is
    // destroyed.
    if (snapshot_future.valid())
      snapshot_future.wait();
  }

  /// Clamps a prefill chunk to the next snapshot or shared-prefix boundary
  /// and marks the state busy.
  TextRunnerPrefill BeginPrefill(std::size_t max_input_tokens) {
    // Requests admitted together share one system prompt: whichever is
    // prefilled first checkpoints it, and the rest start from there.
    if (prefill_offset == 0 && shared_prefix != 0 && InputIdentity().empty() &&
        lease.RestoreSnapshotFor(
            std::span<const TextRunnerToken>(prompt).first(shared_prefix))) {
      prefill_offset = shared_prefix;
      runner->PreparePrefixReuse(
          dynamic_cast<TextRunnerState&>(lease.state()),
          std::span<const TextRunnerToken>(prompt).first(prefill_offset));
    }
    // Stop exactly on the next shared-prefix boundary so it can be
    // persisted. Direct callers join any capture before mutating its
    // borrowed state.
    CapturePromptSnapshot();
    const auto snapshot_position = snapshot_tokens.size();
    if (prefill_offset < snapshot_position)
      max_input_tokens =
          std::min(max_input_tokens, snapshot_position - prefill_offset);
    while (!boundaries.empty() && boundaries.front() <= prefill_offset) {
      boundaries.erase(boundaries.begin());
    }
    if (!boundaries.empty()) {
      max_input_tokens =
          std::min(max_input_tokens, boundaries.front() - prefill_offset);
    }
    state_reusable = false;
    // Complete the model frontier at the cache boundary, including logits
    // and draft catch-up. Its snapshot can then be restored independently.
    return {
        .state = dynamic_cast<TextRunnerState&>(lease.state()),
        .prompt = std::span<const TextRunnerToken>(prompt).first(
            prefill_offset < snapshot_position ? snapshot_position
                                               : prompt.size()),
        .offset = prefill_offset,
        .max_input_tokens = max_input_tokens,
    };
  }

  /// Validates and publishes a chunk started by BeginPrefill.
  TextPrefillStep FinishPrefill(const TextRunnerPrefill& chunk) {
    auto step = chunk.step;
    const std::size_t maximum_consumed =
        std::min(prompt.size() - prefill_offset, chunk.max_input_tokens);
    if (step.consumed_tokens == 0 || step.consumed_tokens > maximum_consumed) {
      throw std::runtime_error(
          "text runner returned an invalid prefill token count");
    }
    prefill_offset += step.consumed_tokens;
    const bool reached_frontier = prefill_offset == prompt.size();
    if (step.decode_ready != (prefill_offset == chunk.prompt.size())) {
      throw std::runtime_error(
          "text runner returned an inconsistent prefill boundary");
    }
    decode_ready = reached_frontier;
    step.decode_ready = reached_frontier;
    state_reusable = true;
    if (!reached_frontier && prefill_offset == snapshot_tokens.size())
      CapturePromptSnapshot(false);
    if (!boundaries.empty() && boundaries.front() == prefill_offset) {
      boundaries.erase(boundaries.begin());
      CaptureSharedPrefix(prefill_offset);
    }
    return step;
  }

  void CapturePromptSnapshot(bool wait = true) noexcept {
    if (!prompt_snapshot_attempted)
      StartPromptSnapshot();
    if (!snapshot_future.valid())
      return;
    if (!wait && snapshot_future.wait_for(std::chrono::seconds(0)) !=
                     std::future_status::ready)
      return;
    FinishPromptSnapshot();
  }

  void StartPromptSnapshot() noexcept {
    if (prompt_snapshot_attempted || prefill_offset != snapshot_tokens.size()) {
      return;
    }
    prompt_snapshot_attempted = true;

    TextRunnerCapabilities capabilities;
    try {
      capabilities = runner->Descriptor().capabilities;
    } catch (...) {
      return;
    }
    if (!capabilities.prefix_reuse || !capabilities.snapshot ||
        !capabilities.fork) {
      return;
    }
    if (lease.cache_hit() && !lease.restored_from_disk() &&
        lease.cached_tokens() == snapshot_tokens.size() &&
        lease.HasSnapshotFor(snapshot_tokens)) {
      return;
    }

    const TextRunnerState* state = nullptr;
    try {
      state = &dynamic_cast<const TextRunnerState&>(lease.state());
    } catch (...) {
      return;
    }
    snapshot_bytes = 0;
    try {
      if (runner->CheckpointPosition(*state) != snapshot_tokens.size()) {
        lease.SkipSnapshot(SnapshotEventReason::kCaptureFailure, 0,
                           snapshot_tokens.size());
        return;
      }
      snapshot_bytes = runner->SnapshotPayloadBytes(*state);
    } catch (...) {
      lease.SkipSnapshot(SnapshotEventReason::kCaptureFailure, 0,
                         snapshot_tokens.size());
      return;
    }

    retain_snapshot = false;
    try {
      retain_snapshot =
          lease.TryReserveSnapshot(snapshot_bytes, snapshot_tokens.size());
    } catch (...) {
      return;
    }
    if (!retain_snapshot) {
      try {
        if (!disk_store)
          return;
        disk_capture = disk_store->ReserveCapture(
            *runner, snapshot_tokens.size(), snapshot_bytes, InputIdentity());
        if (!disk_capture)
          return;
      } catch (...) {
        return;
      }
    }

    snapshot_start = std::chrono::steady_clock::now();
    try {
      snapshot_future = std::async(std::launch::async, [owner = runner, state] {
        return owner->Snapshot(*state);
      });
    } catch (...) {
      disk_capture.reset();
      if (retain_snapshot)
        lease.SkipSnapshot(SnapshotEventReason::kCaptureFailure, snapshot_bytes,
                           snapshot_tokens.size());
    }
  }

  void FinishPromptSnapshot() noexcept {
    try {
      prompt_snapshot = snapshot_future.get();
    } catch (...) {
      prompt_snapshot.reset();
    }
    snapshot_metrics.snapshot_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - snapshot_start)
            .count();
    if (prompt_snapshot == nullptr) {
      disk_capture.reset();
      if (retain_snapshot) {
        lease.SkipSnapshot(SnapshotEventReason::kCaptureFailure, snapshot_bytes,
                           snapshot_tokens.size());
      }
      return;
    }

    if (retain_snapshot && prompt_snapshot->PayloadBytes() != snapshot_bytes) {
      lease.SkipSnapshot(SnapshotEventReason::kReservationMismatch,
                         prompt_snapshot->PayloadBytes(),
                         snapshot_tokens.size());
      retain_snapshot = false;
    }
    // Charge disk-only snapshots immediately. Holding them until commit would
    // let concurrent requests each pass the same queue-capacity preflight.
    const bool already_persisted =
        lease.restored_from_disk() &&
        lease.cached_tokens() == snapshot_tokens.size();
    if (disk_store && !already_persisted) {
      const auto started = std::chrono::steady_clock::now();
      try {
        snapshot_metrics.disk_queued_bytes = disk_store->SaveAsync(
            runner, snapshot_tokens, prompt_snapshot,
            {InputIdentity().begin(), InputIdentity().end()},
            std::move(disk_capture));
      } catch (...) {
        snapshot_metrics.disk_queued_bytes = 0;
      }
      snapshot_metrics.disk_enqueue_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - started)
              .count();
    }
    disk_capture.reset();
    if (!retain_snapshot)
      prompt_snapshot.reset();
  }

  /// Checkpoints the continuation at a shared-prefix boundary reached by
  /// prefill.
  ///
  /// The boundary is the runner's shared prompt prefix (system and tools) or
  /// was learned from stored entries that agree with this prompt up to
  /// `position`, so the next conversation sharing that prefix restores it
  /// instead of prefilling it again. The bounded worker persists the
  /// immutable snapshot without blocking other requests on disk I/O; the
  /// runner's shared prefix is also retained in RAM.
  void CaptureSharedPrefix(std::size_t position) noexcept {
    const bool ram = position == shared_prefix;
    if ((disk_store == nullptr && !ram) || position == 0 ||
        position >= prompt.size()) {
      return;
    }
    const auto start = std::chrono::steady_clock::now();
    bool failed = false;
    std::size_t reserved = 0;
    try {
      const auto capabilities = runner->Descriptor().capabilities;
      if (!capabilities.prefix_reuse || !capabilities.snapshot ||
          !capabilities.fork) {
        return;
      }
      const auto& state = dynamic_cast<const TextRunnerState&>(lease.state());
      if (runner->CheckpointPosition(state) != position) {
        return;
      }
      const auto prefix =
          std::span<const TextRunnerToken>(prompt).first(position);
      const auto bytes = runner->SnapshotPayloadBytes(state);
      // Another request may have stored this prefix meanwhile.
      const bool to_disk =
          disk_store != nullptr &&
          !disk_store->Touch(*runner, prefix, InputIdentity()) &&
          disk_store->CanSave(*runner, position, bytes, InputIdentity());
      // A pending prompt snapshot owns the lease's single reservation.
      if (ram && !snapshot_future.valid() && !lease.HasSnapshotFor(prefix) &&
          lease.TryReserveSnapshot(bytes, position, true))
        reserved = bytes;
      if (!to_disk && reserved == 0)
        return;
      std::shared_ptr<const TextRunnerSnapshot> snapshot =
          runner->Snapshot(state);
      if (snapshot == nullptr) {
        throw std::runtime_error("shared prefix capture failed");
      }
      std::size_t saved = 0;
      if (reserved != 0) {
        reserved = 0;
        saved +=
            lease.PublishSnapshot({prefix.begin(), prefix.end()}, snapshot);
      }
      if (to_disk) {
        saved += disk_store->SaveAsync(
            runner, {prefix.begin(), prefix.end()}, std::move(snapshot),
            {InputIdentity().begin(), InputIdentity().end()});
      }
      if (saved != 0) {
        ++snapshot_metrics.shared_prefix_snapshots;
        snapshot_metrics.shared_prefix_bytes += saved;
      }
    } catch (...) {
      // Best effort: the request itself is unaffected.
      failed = true;
      if (reserved != 0)
        lease.SkipSnapshot(SnapshotEventReason::kCaptureFailure, reserved,
                           position);
    }
    snapshot_metrics.shared_prefix_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start)
            .count();
    if (failed) {
      snapshot_metrics.shared_prefix_failures += 1;
    }
  }

  std::shared_ptr<TextModelRunner> runner;
  std::shared_ptr<ContinuationDiskStore> disk_store;
  ContinuationCache::Lease lease;
  std::vector<TextRunnerToken> prompt;
  std::vector<TextRunnerToken> snapshot_tokens;
  /// Ascending prefill positions to persist, all inside (cached, prompt size).
  std::vector<std::size_t> boundaries;
  /// Runner's shared prompt prefix; its checkpoint is also kept in RAM.
  std::size_t shared_prefix{0};
  std::vector<TextRunnerToken> generated;
  std::size_t prefill_offset{0};
  bool decode_ready{false};
  // A throwing model call may have partially mutated device state. Never
  // publish that state as a live continuation, even if its position is stale.
  bool state_reusable{true};
  bool stopped{false};
  std::optional<TextDecodeSelection> pending_selection;
  sampling::SamplerState sampler;
  std::shared_ptr<const TextPromptContext> context;
  [[nodiscard]] std::span<const std::uint8_t> InputIdentity() const {
    return context ? std::span<const std::uint8_t>(context->cache_identity)
                   : std::span<const std::uint8_t>{};
  }
  bool prompt_snapshot_attempted{false};
  std::shared_ptr<const TextRunnerSnapshot> prompt_snapshot;
  TextRunnerPool::Request::CommitMetrics snapshot_metrics;
  std::size_t snapshot_bytes{0};
  bool retain_snapshot{false};
  std::chrono::steady_clock::time_point snapshot_start;
  std::unique_ptr<ContinuationDiskStore::CaptureReservation> disk_capture;
  std::future<std::unique_ptr<TextRunnerSnapshot>> snapshot_future;
};

TextRunnerPool::Request::Request() = default;

TextRunnerPool::Request::Request(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

void TextRunnerPool::Request::CapturePromptSnapshot() {
  if (impl_)
    impl_->CapturePromptSnapshot();
}

bool TextRunnerPool::Request::PreparePromptSnapshot() {
  if (impl_)
    impl_->CapturePromptSnapshot(false);
  return !impl_ || !impl_->snapshot_future.valid();
}

bool TextRunnerPool::Request::SnapshotPending() const {
  return impl_ && impl_->snapshot_future.valid() &&
         impl_->snapshot_future.wait_for(std::chrono::seconds(0)) !=
             std::future_status::ready;
}

std::optional<TextDecodeSelection>
TextRunnerPool::Request::PreviewFirstToken() {
  if (!impl_ || !impl_->decode_ready || !impl_->generated.empty() ||
      impl_->pending_selection || impl_->stopped)
    return std::nullopt;
  auto sampler = impl_->sampler;
  return impl_->runner->PreviewFirstToken(
      dynamic_cast<TextRunnerState&>(impl_->lease.state()), sampler);
}

TextRunnerPool::Request::~Request() = default;

TextRunnerPool::Request::Request(Request&&) noexcept = default;

TextRunnerPool::Request& TextRunnerPool::Request::operator=(
    Request&&) noexcept = default;

TextRunnerPool::Request::operator bool() const noexcept {
  return impl_ != nullptr && static_cast<bool>(impl_->lease);
}

bool TextRunnerPool::Request::cache_hit() const noexcept {
  return impl_ != nullptr && impl_->lease.cache_hit();
}

std::size_t TextRunnerPool::Request::cached_prompt_tokens() const noexcept {
  return impl_ != nullptr ? impl_->lease.cached_tokens() : 0;
}

ContinuationLookup TextRunnerPool::Request::cache_lookup() const noexcept {
  return impl_ != nullptr ? impl_->lease.lookup() : ContinuationLookup{};
}

std::size_t TextRunnerPool::Request::cache_restore_bytes() const noexcept {
  return impl_ != nullptr ? impl_->lease.restored_snapshot_bytes() : 0;
}

double TextRunnerPool::Request::cache_restore_ms() const noexcept {
  return impl_ != nullptr ? impl_->lease.restore_ms() : 0.0;
}

bool TextRunnerPool::Request::cache_disk_hit() const noexcept {
  return impl_ != nullptr && impl_->lease.restored_from_disk();
}

std::size_t TextRunnerPool::Request::prompt_tokens() const noexcept {
  return impl_ != nullptr ? impl_->prompt.size() : 0;
}

bool TextRunnerPool::Request::prefill_complete() const noexcept {
  return impl_ != nullptr && impl_->decode_ready;
}

void TextRunnerPool::Request::PrepareBatchExecution() {
  if (!*this) {
    throw std::logic_error("text runner request is empty");
  }
  impl_->runner->PrepareBatchExecution(
      dynamic_cast<TextRunnerState&>(impl_->lease.state()));
}

TextPrefillStep TextRunnerPool::Request::Prefill(std::size_t max_input_tokens) {
  if (!*this) {
    throw std::logic_error("text runner request is empty");
  }
  if (impl_->pending_selection.has_value()) {
    throw std::logic_error(
        "text runner cannot prefill with a pending decode token");
  }
  if (impl_->stopped) {
    throw std::logic_error("text runner request already stopped");
  }
  if (impl_->decode_ready) {
    return {
        .consumed_tokens = 0,
        .decode_ready = true,
    };
  }
  if (max_input_tokens == 0) {
    throw std::invalid_argument(
        "text runner prefill budget must be at least one token");
  }

  auto chunk = impl_->BeginPrefill(max_input_tokens);
  chunk.step = impl_->runner->Prefill(chunk.state, chunk.prompt, chunk.offset,
                                      chunk.max_input_tokens);
  return impl_->FinishPrefill(chunk);
}

TextDecodeSelection TextRunnerPool::Request::SelectNext() {
  if (!*this) {
    throw std::logic_error("text runner request is empty");
  }
  if (!impl_->decode_ready) {
    throw std::logic_error(
        "text runner cannot decode before prefill completes");
  }
  if (impl_->pending_selection.has_value()) {
    throw std::logic_error(
        "text runner decode token must be advanced before selecting another");
  }
  if (impl_->stopped) {
    throw std::logic_error("text runner request already stopped");
  }

  impl_->state_reusable = false;
  auto selection = impl_->runner->SelectNext(
      dynamic_cast<TextRunnerState&>(impl_->lease.state()), impl_->sampler);
  selection.logprobs = impl_->sampler.TakeLogprobs();
  impl_->state_reusable = true;
  if (selection.stop) {
    impl_->stopped = true;
    return selection;
  }
  CommitToken(impl_->sampler, selection.token);
  impl_->generated.push_back(selection.token);
  impl_->pending_selection = selection;
  return selection;
}

void TextRunnerPool::Request::Advance() {
  if (!*this) {
    throw std::logic_error("text runner request is empty");
  }
  if (!impl_->pending_selection.has_value()) {
    throw std::logic_error(
        "text runner request has no selected token to advance");
  }
  impl_->CapturePromptSnapshot();
  impl_->state_reusable = false;
  impl_->runner->Advance(dynamic_cast<TextRunnerState&>(impl_->lease.state()),
                         impl_->pending_selection->token);
  impl_->pending_selection.reset();
  impl_->state_reusable = true;
}

TextDecodeStep TextRunnerPool::Request::DecodeStep(std::size_t max_tokens) {
  if (!*this) {
    throw std::logic_error("text runner request is empty");
  }
  if (!impl_->decode_ready) {
    throw std::logic_error(
        "text runner cannot decode before prefill completes");
  }
  if (impl_->pending_selection.has_value()) {
    throw std::logic_error(
        "text runner cannot decode a step with a pending token");
  }
  if (impl_->stopped) {
    throw std::logic_error("text runner request already stopped");
  }
  if (max_tokens == 0) {
    throw std::invalid_argument(
        "text runner decode step budget must be at least one token");
  }

  impl_->CapturePromptSnapshot();
  impl_->state_reusable = false;
  auto step = impl_->runner->DecodeStep(
      dynamic_cast<TextRunnerState&>(impl_->lease.state()), max_tokens,
      impl_->sampler);
  if (step.selections.size() > max_tokens ||
      (step.selections.empty() && !step.stop)) {
    throw std::runtime_error("text runner returned an invalid decode step");
  }
  for (const auto& selection : step.selections) {
    if (selection.stop) {
      throw std::runtime_error(
          "text runner decode step contains an embedded stop selection");
    }
    CommitToken(impl_->sampler, selection.token);
    impl_->generated.push_back(selection.token);
  }
  impl_->stopped = step.stop;
  impl_->state_reusable = true;
  return step;
}

TextRunnerPool::Request::CommitMetrics TextRunnerPool::Request::Commit() {
  if (!*this) {
    throw std::logic_error("text runner request is empty");
  }
  if (!impl_->decode_ready) {
    throw std::logic_error(
        "text runner cannot commit before prefill completes");
  }
  if (impl_->pending_selection.has_value()) {
    if (impl_->runner->Descriptor().capabilities.final_token_advance_required) {
      throw std::logic_error(
          "text runner cannot commit an unadvanced decode token");
    }
    impl_->pending_selection.reset();
  }

  impl_->CapturePromptSnapshot();
  auto& state = dynamic_cast<TextRunnerState&>(impl_->lease.state());
  state.SetCancellationCheck({});
  if (!impl_->runner->Descriptor().capabilities.prefix_reuse) {
    impl_->lease.Invalidate();
    impl_.reset();
    return {};
  }
  const auto capabilities = impl_->runner->Descriptor().capabilities;
  std::vector<ContinuationToken> checkpoint = impl_->prompt;
  checkpoint.insert(checkpoint.end(), impl_->generated.begin(),
                    impl_->generated.end());
  const std::size_t position = impl_->runner->CheckpointPosition(state);
  if (position < impl_->lease.cached_tokens() || position > checkpoint.size()) {
    throw std::runtime_error(
        "text runner checkpoint is outside executed token history");
  }
  checkpoint.resize(position);
  if (capabilities.snapshot && capabilities.fork) {
    CommitMetrics metrics = impl_->snapshot_metrics;
    metrics.snapshot_bytes = impl_->lease.Commit(
        std::move(impl_->snapshot_tokens), std::move(impl_->prompt_snapshot),
        std::move(checkpoint));
    impl_.reset();
    return metrics;
  }

  std::unique_ptr<TextRunnerSnapshot> snapshot;
  CommitMetrics metrics;
  metrics.snapshot_bytes =
      impl_->lease.Commit(std::move(checkpoint), std::move(snapshot));
  impl_.reset();
  return metrics;
}

TextRunnerPool::Request::CommitMetrics
TextRunnerPool::Request::Cancel() noexcept {
  if (!*this)
    return {};
  try {
    // Join only an existing capture; cancellation must not start a new copy
    // or execute the pending selected token merely to retain the session.
    if (impl_->snapshot_future.valid())
      impl_->FinishPromptSnapshot();
    auto& state = dynamic_cast<TextRunnerState&>(impl_->lease.state());
    state.SetCancellationCheck({});
    const auto capabilities = impl_->runner->Descriptor().capabilities;
    if (!capabilities.prefix_reuse) {
      Invalidate();
      return {};
    }
    std::vector<ContinuationToken> checkpoint;
    if (impl_->state_reusable &&
        (impl_->decode_ready ||
         impl_->prefill_offset == impl_->snapshot_tokens.size())) {
      if (impl_->decode_ready)
        impl_->runner->PrepareCancellation(state);
      checkpoint.assign(impl_->prompt.begin(),
                        impl_->prompt.begin() + impl_->prefill_offset);
      if (impl_->decode_ready)
        checkpoint.insert(checkpoint.end(), impl_->generated.begin(),
                          impl_->generated.end());
      const auto position = impl_->runner->CheckpointPosition(state);
      if (position < impl_->lease.cached_tokens() ||
          position > checkpoint.size())
        throw std::runtime_error(
            "cancelled checkpoint is outside completed work");
      checkpoint.resize(position);
    } else {
      state.Invalidate();
    }
    CommitMetrics metrics = impl_->snapshot_metrics;
    if (capabilities.snapshot && capabilities.fork) {
      metrics.snapshot_bytes = impl_->lease.Commit(
          std::move(impl_->snapshot_tokens), std::move(impl_->prompt_snapshot),
          std::move(checkpoint));
    } else {
      metrics.snapshot_bytes = impl_->lease.Commit(std::move(checkpoint));
    }
    impl_.reset();
    return metrics;
  } catch (...) {
    Invalidate();
    return {};
  }
}

void TextRunnerPool::Request::Invalidate() noexcept {
  if (impl_ != nullptr) {
    if (impl_->snapshot_future.valid())
      impl_->snapshot_future.wait();
    dynamic_cast<TextRunnerState&>(impl_->lease.state())
        .SetCancellationCheck({});
    impl_->lease.Invalidate();
    impl_.reset();
  }
}

TextRunnerPool::TextRunnerPool(
    std::shared_ptr<TextModelRunner> runner, std::size_t state_count,
    std::optional<TextRunnerDiskCacheOptions> disk_cache)
    : impl_(std::make_unique<Impl>(std::move(runner), state_count,
                                   std::move(disk_cache))) {}

TextRunnerPool::~TextRunnerPool() = default;

const TextModelRunner& TextRunnerPool::runner() const noexcept {
  return *impl_->validated.runner;
}

std::size_t TextRunnerPool::capacity() const noexcept {
  return impl_->cache.capacity();
}

TextRunnerPool::TextRunnerPoolMetrics TextRunnerPool::metrics_snapshot() const {
  TextRunnerPoolMetrics metrics{
      .host_snapshot_bytes = impl_->cache.retained_snapshot_bytes(),
      .host_snapshot_capacity_bytes = impl_->cache.snapshot_capacity_bytes(),
  };
  if (impl_->disk_store != nullptr) {
    metrics.disk_cache_bytes = impl_->disk_store->retained_bytes();
    metrics.disk_cache_capacity_bytes = impl_->disk_store->capacity_bytes();
    metrics.disk_cache_entries = impl_->disk_store->entry_count();
  }
  return metrics;
}

TextExecutionPlan TextRunnerPool::SelectDecodePlan(
    std::size_t ready_requests) const {
  if (ready_requests == 0) {
    throw std::invalid_argument(
        "decode plan requires at least one ready request");
  }

  const auto serial =
      std::find_if(impl_->validated.plans.begin(), impl_->validated.plans.end(),
                   [](const TextExecutionPlan& plan) {
                     return plan.kind == TextExecutionPlanKind::kSerial &&
                            plan.physical_width == 1;
                   });
  if (ready_requests == 1) {
    return *serial;
  }

  const TextExecutionPlan* smallest_covering = nullptr;
  const TextExecutionPlan* widest_available = nullptr;
  for (const auto& plan : impl_->validated.plans) {
    if (plan.kind != TextExecutionPlanKind::kBatched) {
      continue;
    }
    if (widest_available == nullptr ||
        plan.physical_width > widest_available->physical_width) {
      widest_available = &plan;
    }
    if (plan.physical_width >= ready_requests &&
        (smallest_covering == nullptr ||
         plan.physical_width < smallest_covering->physical_width)) {
      smallest_covering = &plan;
    }
  }
  if (smallest_covering != nullptr) {
    return *smallest_covering;
  }
  if (widest_available != nullptr) {
    return *widest_available;
  }
  return *serial;
}

std::vector<TextRunnerPrefill> TextRunnerPool::BeginPrefills(
    std::span<Request* const> requests,
    std::span<PrefillCompanion> prefills) const {
  std::vector<TextRunnerPrefill> chunks;
  chunks.reserve(prefills.size());
  for (std::size_t index = 0; index < prefills.size(); ++index) {
    auto* request = prefills[index].request;
    const auto previous = prefills.first(index);
    if (request == nullptr || !*request ||
        request->impl_->runner != impl_->validated.runner ||
        request->impl_->decode_ready || request->impl_->stopped ||
        request->impl_->pending_selection.has_value() ||
        prefills[index].max_input_tokens == 0 ||
        std::find(requests.begin(), requests.end(), request) !=
            requests.end() ||
        std::any_of(previous.begin(), previous.end(), [&](const auto& other) {
          return other.request == request;
        })) {
      throw std::logic_error("invalid prefill request in text batch");
    }
  }
  for (auto& prefill : prefills)
    chunks.push_back(
        prefill.request->impl_->BeginPrefill(prefill.max_input_tokens));
  return chunks;
}

void TextRunnerPool::FinishPrefills(std::span<PrefillCompanion> prefills,
                                    std::span<TextRunnerPrefill> chunks) {
  for (std::size_t index = 0; index < prefills.size(); ++index) {
    auto& prefill = prefills[index];
    prefill.failure = chunks[index].failure;
    if (prefill.failure)
      continue;
    try {
      prefill.step = prefill.request->impl_->FinishPrefill(chunks[index]);
    } catch (...) {
      prefill.failure = std::current_exception();
    }
  }
}

void TextRunnerPool::PrefillBatch(std::span<PrefillCompanion> prefills) {
  if (prefills.empty())
    return;
  auto chunks = BeginPrefills({}, prefills);
  (void)impl_->validated.runner->DecodePrefillBatch({}, chunks);
  FinishPrefills(prefills, chunks);
}

std::vector<std::exception_ptr> TextRunnerPool::AdvanceBatch(
    std::span<Request*> requests, const TextExecutionPlan& plan,
    std::span<PrefillCompanion> prefills) {
  const std::size_t width = requests.size() + prefills.size();
  if (plan.kind != TextExecutionPlanKind::kBatched || requests.empty() ||
      width < 2 || width > plan.physical_width) {
    throw std::invalid_argument("invalid batched text execution plan");
  }
  if (std::find(impl_->validated.plans.begin(), impl_->validated.plans.end(),
                plan) == impl_->validated.plans.end()) {
    throw std::invalid_argument(
        "text runner does not support the requested batched plan");
  }

  std::vector<std::exception_ptr> failures(requests.size());
  std::vector<TextRunnerAdvance> advances;
  advances.reserve(requests.size());
  for (std::size_t index = 0; index < requests.size(); ++index) {
    auto* request = requests[index];
    if (request == nullptr || !*request ||
        request->impl_->runner != impl_->validated.runner ||
        !request->impl_->pending_selection.has_value()) {
      throw std::logic_error("invalid request in batched text advance");
    }
    const auto previous_requests = requests.first(index);
    if (std::find(previous_requests.begin(), previous_requests.end(),
                  request) != previous_requests.end()) {
      throw std::invalid_argument(
          "batched text advance contains a duplicate request");
    }
    advances.push_back({
        .state = dynamic_cast<TextRunnerState&>(request->impl_->lease.state()),
        .token = request->impl_->pending_selection->token,
        .failure = &failures[index],
    });
  }

  for (auto* request : requests) {
    request->CapturePromptSnapshot();
    request->impl_->state_reusable = false;
  }
  auto chunks = BeginPrefills(requests, prefills);
  if (chunks.empty())
    impl_->validated.runner->AdvanceBatch(advances);
  else
    impl_->validated.runner->AdvancePrefillBatch(advances, chunks);
  FinishPrefills(prefills, chunks);
  for (std::size_t i = 0; i < requests.size(); ++i) {
    if (!failures[i]) {
      requests[i]->impl_->pending_selection.reset();
      requests[i]->impl_->state_reusable = true;
    }
  }
  return failures;
}

std::vector<TextDecodeStep> TextRunnerPool::DecodeBatch(
    std::span<Request*> requests, std::span<const std::size_t> max_tokens,
    const TextExecutionPlan& plan, std::span<PrefillCompanion> prefills) {
  const std::size_t width = requests.size() + prefills.size();
  if (plan.kind != TextExecutionPlanKind::kBatched || requests.empty() ||
      width < 2 || width > plan.physical_width ||
      max_tokens.size() != requests.size()) {
    throw std::invalid_argument("invalid batched text decode plan");
  }
  if (std::find(impl_->validated.plans.begin(), impl_->validated.plans.end(),
                plan) == impl_->validated.plans.end()) {
    throw std::invalid_argument(
        "text runner does not support the requested batched plan");
  }

  std::vector<TextRunnerDecode> decodes;
  decodes.reserve(requests.size());
  for (std::size_t index = 0; index < requests.size(); ++index) {
    auto* request = requests[index];
    if (request == nullptr || !*request ||
        request->impl_->runner != impl_->validated.runner ||
        !request->impl_->decode_ready || request->impl_->stopped ||
        request->impl_->pending_selection.has_value() ||
        max_tokens[index] == 0) {
      throw std::logic_error("invalid request in batched text decode");
    }
    const auto previous_requests = requests.first(index);
    if (std::find(previous_requests.begin(), previous_requests.end(),
                  request) != previous_requests.end()) {
      throw std::invalid_argument(
          "batched text decode contains a duplicate request");
    }
    decodes.push_back({
        .state = dynamic_cast<TextRunnerState&>(request->impl_->lease.state()),
        .max_tokens = max_tokens[index],
        .sampler = request->impl_->sampler,
    });
  }

  for (auto* request : requests) {
    request->CapturePromptSnapshot();
    request->impl_->state_reusable = false;
  }
  auto chunks = BeginPrefills(requests, prefills);
  auto steps = chunks.empty() ? impl_->validated.runner->DecodeBatch(decodes)
                              : impl_->validated.runner->DecodePrefillBatch(
                                    decodes, chunks);
  FinishPrefills(prefills, chunks);
  if (steps.size() != requests.size()) {
    throw std::runtime_error(
        "text runner returned an invalid decode batch size");
  }
  for (std::size_t index = 0; index < requests.size(); ++index) {
    auto& step = steps[index];
    if (step.failure)
      continue;
    auto& request = *requests[index]->impl_;
    if (step.execution_plan.physical_width == 0 ||
        step.execution_plan.physical_width > requests.size() ||
        (step.execution_plan.kind == TextExecutionPlanKind::kSerial &&
         step.execution_plan.physical_width != 1)) {
      throw std::runtime_error(
          "text runner returned an invalid execution width");
    }
    if (step.selections.size() > max_tokens[index] ||
        (step.selections.empty() && !step.stop)) {
      throw std::runtime_error(
          "text runner returned an invalid batched decode step");
    }
    for (const auto& selection : step.selections) {
      if (selection.stop) {
        throw std::runtime_error(
            "text runner batched decode contains an embedded stop selection");
      }
      CommitToken(request.sampler, selection.token);
      request.generated.push_back(selection.token);
    }
    request.stopped = step.stop;
    request.state_reusable = true;
  }
  return steps;
}

TextRunnerPool::Request TextRunnerPool::Acquire(
    std::vector<TextRunnerToken> prompt,
    const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled,
    std::shared_ptr<const TextPromptContext> context, bool reuse_prompt,
    std::size_t cache_prefix_tokens) {
  sampling_config.Validate();
  if (prompt.empty()) {
    throw std::invalid_argument("text runner prompt must not be empty");
  }
  if (prompt.size() > impl_->validated.descriptor.max_context) {
    throw std::length_error("text runner prompt exceeds model context");
  }
  if (cache_prefix_tokens > prompt.size())
    throw std::invalid_argument("cache prefix exceeds prompt length");

  std::unique_lock admission(impl_->admission_mutex, std::defer_lock);
  while (!admission.try_lock_for(std::chrono::milliseconds(10))) {
    if (is_cancelled && is_cancelled())
      return {};
  }

  const std::span<const std::uint8_t> identity =
      context ? std::span<const std::uint8_t>(context->cache_identity)
              : std::span<const std::uint8_t>{};
  // Older disk entries may include the mutable assistant-generation suffix.
  // Restore only through the stable boundary so this request can retain a
  // checkpoint that also survives interrupted or reformatted assistant turns.
  const auto reusable = std::span<const TextRunnerToken>(prompt).first(
      cache_prefix_tokens == 0 ? prompt.size() : cache_prefix_tokens);
  const auto cancelled = [&is_cancelled] {
    return is_cancelled && is_cancelled();
  };
  auto& runner = *impl_->validated.runner;
  // A RAM checkpoint much shorter than the longest one on disk would shadow
  // it (the shared system prompt over a conversation's own history).
  // ponytail: fixed margin, roughly the prefill a disk read of the longer
  // checkpoint costs; derive from restore_ms and prefill rate if it matters.
  constexpr std::size_t kDiskPreferenceTokens = 2048;
  std::size_t min_ram_tokens = 0;
  if (reuse_prompt && impl_->disk_store != nullptr) {
    try {
      const auto on_disk =
          impl_->disk_store->LongestPrefixTokens(runner, reusable, identity);
      if (on_disk > kDiskPreferenceTokens)
        min_ram_tokens = on_disk - kDiskPreferenceTokens;
    } catch (...) {
      // RAM lookup alone remains correct.
    }
  }
  auto lease = impl_->cache.Acquire(
      reusable, is_cancelled, identity,
      [&](ContinuationState& state) {
        auto& text_state = dynamic_cast<TextRunnerState&>(state);
        text_state.SetCancellationCheck(is_cancelled);
        runner.SetPromptContext(text_state, context);
      },
      reuse_prompt, min_ram_tokens);
  if (!lease) {
    return {};
  }
  const bool ram_checkpoints =
      impl_->cache.capacity() > 1 &&
      impl_->validated.descriptor.capabilities.snapshot &&
      impl_->validated.descriptor.capabilities.fork;
  // Retains the leased state (at prefix) as an immutable RAM snapshot that
  // peers restore instead of prefilling, and on disk when `persist`.
  const auto checkpoint = [&](std::span<const TextRunnerToken> prefix,
                              bool persist, std::string_view event) {
    auto& state = dynamic_cast<TextRunnerState&>(lease.state());
    std::size_t bytes = 0;
    std::shared_ptr<const TextRunnerSnapshot> snapshot;
    const auto started = std::chrono::steady_clock::now();
    try {
      if (runner.CheckpointPosition(state) != prefix.size())
        throw std::logic_error("live checkpoint position mismatch");
      bytes = runner.SnapshotPayloadBytes(state);
      if (lease.TryReserveSnapshot(bytes, prefix.size(), true)) {
        snapshot = runner.Snapshot(state);
        if (snapshot == nullptr)
          throw std::runtime_error("live checkpoint capture failed");
        if (lease.PublishSnapshot({prefix.begin(), prefix.end()}, snapshot) ==
            0)
          snapshot.reset();
      }
    } catch (...) {
      snapshot.reset();
      lease.SkipSnapshot(SnapshotEventReason::kCaptureFailure, bytes,
                         prefix.size());
    }
    if (!snapshot)
      return;
    if (persist && impl_->disk_store) {
      try {
        (void)impl_->disk_store->SaveAsync(
            impl_->validated.runner, {prefix.begin(), prefix.end()}, snapshot,
            {identity.begin(), identity.end()});
      } catch (...) {
        // RAM branching does not depend on optional disk admission.
      }
    }
    std::ostringstream line;
    line << "event=" << event << " tokens=" << prefix.size()
         << " bytes=" << bytes << " capture_ms="
         << std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started)
                .count();
    Logger::Info("cache", line.str());
  };
  // A live frontier can contain generated tokens beyond the prompt snapshot.
  // Freeze it before the first branch mutates it, so peers restore the same
  // decode history instead of feeding generated output through prefill.
  // C1 keeps its copy-free live path and its original branching checkpoint.
  if (ram_checkpoints && lease.cache_hit() && !cancelled()) {
    const auto prefix = reusable.first(lease.cached_tokens());
    if (!lease.HasSnapshotFor(prefix))
      checkpoint(prefix, true, "live_checkpoint");
  }
  if (reuse_prompt && !lease.cache_hit() && impl_->disk_store != nullptr &&
      !cancelled()) {
    const auto restore_start = std::chrono::steady_clock::now();
    try {
      const auto restored = impl_->disk_store->RestoreLongestPrefix(
          runner, dynamic_cast<TextRunnerState&>(lease.state()), reusable,
          identity);
      if (restored.restored) {
        lease.AdoptRestoredPrefix(
            restored.token_count, restored.file_bytes,
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - restore_start)
                .count());
      } else {
        auto& state = dynamic_cast<TextRunnerState&>(lease.state());
        state.SetCancellationCheck(is_cancelled);
        runner.SetPromptContext(state, context);
      }
    } catch (...) {
      lease.state().Invalidate();
      auto& state = dynamic_cast<TextRunnerState&>(lease.state());
      state.SetCancellationCheck(is_cancelled);
      runner.SetPromptContext(state, context);
    }
  }
  std::size_t shared_prefix = 0;
  if (reuse_prompt && !cancelled()) {
    try {
      shared_prefix = runner.SharedPromptPrefix(prompt);
    } catch (...) {
      shared_prefix = 0;
    }
    if (shared_prefix < impl_->shared_prefix_min_tokens ||
        shared_prefix >= reusable.size())
      shared_prefix = 0;
  }
  if (shared_prefix != 0) {
    const auto prefix = reusable.first(shared_prefix);
    // The next conversation finds a disk-restored shared prefix in RAM.
    if (ram_checkpoints && lease.restored_from_disk() &&
        lease.cached_tokens() == shared_prefix && !lease.HasSnapshotFor(prefix))
      checkpoint(prefix, false, "shared_prefix_checkpoint");
    impl_->ObserveSharedPrefix(prompt, shared_prefix, identity);
  }
  std::vector<std::size_t> boundaries;
  if (reuse_prompt && impl_->disk_store != nullptr && !cancelled()) {
    try {
      boundaries = impl_->disk_store->SharedPrefixBoundaries(
          runner, prompt, impl_->shared_prefix_min_tokens,
          impl_->shared_prefix_max_boundaries, identity);
    } catch (...) {
      boundaries.clear();
    }
  }
  if (shared_prefix != 0 && (ram_checkpoints || impl_->disk_store)) {
    boundaries.insert(std::ranges::lower_bound(boundaries, shared_prefix),
                      shared_prefix);
    const auto [first, last] = std::ranges::unique(boundaries);
    boundaries.erase(first, last);
  }
  std::erase_if(boundaries, [&lease](std::size_t boundary) {
    return boundary <= lease.cached_tokens();
  });
  return Request(std::make_unique<Request::Impl>(
      impl_->validated.runner, impl_->disk_store, std::move(lease),
      std::move(prompt), std::move(boundaries), sampling_config,
      std::move(context), cache_prefix_tokens,
      ram_checkpoints ? shared_prefix : 0));
}

void TextRunnerPool::Prefetch(
    std::span<const TextRunnerToken> prompt,
    const std::shared_ptr<const TextPromptContext>& context, bool reuse_prompt,
    std::size_t cache_prefix_tokens) noexcept {
  if (!reuse_prompt || impl_->disk_store == nullptr || prompt.empty() ||
      cache_prefix_tokens > prompt.size())
    return;
  try {
    const std::span<const std::uint8_t> identity =
        context ? std::span<const std::uint8_t>(context->cache_identity)
                : std::span<const std::uint8_t>{};
    const auto reusable = prompt.first(
        cache_prefix_tokens == 0 ? prompt.size() : cache_prefix_tokens);
    auto& runner = *impl_->validated.runner;
    // Acquire prefers a disk checkpoint over a RAM one that is more than
    // kDiskPreferenceTokens shorter.
    constexpr std::size_t kDiskPreferenceTokens = 2048;
    const auto on_disk =
        impl_->disk_store->LongestPrefixTokens(runner, reusable, identity);
    if (on_disk == 0 || impl_->cache.LongestCachedTokens(reusable, identity) +
                                kDiskPreferenceTokens >=
                            on_disk)
      return;
    impl_->disk_store->PrefetchLongestPrefix(runner, reusable, identity);
  } catch (...) {
    // A missed read-ahead only costs the inline read at admission.
  }
}

TextRunnerPool::Request TextRunnerPool::Acquire(
    std::vector<TextRunnerToken> prompt,
    const CancellationCheck& is_cancelled) {
  return Acquire(std::move(prompt), sampling::SamplingConfig{}, is_cancelled);
}

}  // namespace gufo::server

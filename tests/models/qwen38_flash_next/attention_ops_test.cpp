#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

// The model's full-attention geometry: 24 query heads over 2 KV heads,
// head_dim 256, 4-token selection blocks.
constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kDim = 256;
constexpr std::uint32_t kRatio = 4;
constexpr std::uint32_t kQWidth = kHeads * kDim;
constexpr std::uint32_t kKvWidth = kKvHeads * kDim;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

template<typename T>
class HipBuffer {
public:
  explicit HipBuffer(std::size_t count) : count_(count) {
    void* allocation = nullptr;
    CheckHip(hipMalloc(&allocation, bytes()), "hipMalloc");
    CheckHip(hipMemset(allocation, 0, bytes()), "hipMemset");
    data_ = static_cast<T*>(allocation);
  }
  ~HipBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }

  HipBuffer(const HipBuffer&) = delete;
  HipBuffer& operator=(const HipBuffer&) = delete;
  HipBuffer(HipBuffer&&) = delete;
  HipBuffer& operator=(HipBuffer&&) = delete;

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return count_ * sizeof(T);
  }

private:
  T* data_{nullptr};
  std::size_t count_{0};
};

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

std::vector<float> MakeValues(std::size_t count, std::uint32_t seed,
                              float scale) {
  std::vector<float> values(count);
  for (float& value : values) {
    value = scale *
            static_cast<float>(static_cast<int>(NextRandom(&seed) & 0xFFFFU) -
                               32768) /
            32768.0F;
  }
  return values;
}

template<typename T>
void Upload(HipBuffer<T>* destination, const std::vector<T>& source) {
  CheckHip(hipMemcpy(destination->get(), source.data(), destination->bytes(),
                     hipMemcpyHostToDevice),
           "upload");
}

std::vector<float> Download(HipBuffer<float>* source, std::size_t count) {
  std::vector<float> values(count);
  CheckHip(hipMemcpy(values.data(), source->get(), source->bytes(),
                     hipMemcpyDeviceToHost),
           "download");
  return values;
}

void CheckPreparation(std::uint32_t n, std::uint32_t start,
                      std::uint32_t rotary) {
  constexpr std::uint32_t stride = 2 * (kQWidth + kKvWidth);
  const std::size_t count = static_cast<std::size_t>(n) * kQWidth;
  const std::size_t cache_rows = start + n + 2;
  HipBuffer<float> packed(static_cast<std::size_t>(n) * stride);
  HipBuffer<float> q_gamma(kDim), k_gamma(kDim);
  HipBuffer<float> q_ref(count), gate_ref(count), q_out(count), gate_out(count);
  HipBuffer<float> k(static_cast<std::size_t>(n) * kKvWidth);
  HipBuffer<float> v(static_cast<std::size_t>(n) * kKvWidth);
  HipBuffer<__half> k_ref(cache_rows * kKvWidth), v_ref(cache_rows * kKvWidth);
  HipBuffer<__half> k_out(cache_rows * kKvWidth), v_out(cache_rows * kKvWidth);
  HipBuffer<std::uint32_t> pos(1);
  Upload(&packed, MakeValues(static_cast<std::size_t>(n) * stride, 17, 4.0F));
  auto qg = MakeValues(kDim, 37, 0.5F);
  auto kg = MakeValues(kDim, 91, 0.5F);
  for (auto& x : qg)
    x += 1.0F;
  for (auto& x : kg)
    x += 1.0F;
  Upload(&q_gamma, qg);
  Upload(&k_gamma, kg);

  struct Capture {
    hipStream_t stream{nullptr};
    hipGraph_t graph{nullptr};
    hipGraphExec_t exec{nullptr};
    Capture() {
      CheckHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
               "preparation stream");
    }
    ~Capture() {
      if (exec)
        (void)hipGraphExecDestroy(exec);
      if (graph)
        (void)hipGraphDestroy(graph);
      (void)hipStreamDestroy(stream);
    }
  } capture;
  const auto same = [&](auto* expected, auto* actual, std::size_t size,
                        const char* name, std::uint32_t replay) {
    using T = std::remove_pointer_t<decltype(expected)>;
    std::vector<T> a(size), b(size);
    CheckHip(
        hipMemcpy(a.data(), expected, size * sizeof(T), hipMemcpyDeviceToHost),
        "preparation reference");
    CheckHip(
        hipMemcpy(b.data(), actual, size * sizeof(T), hipMemcpyDeviceToHost),
        "preparation result");
    for (std::size_t i = 0; i < size; ++i) {
      if (std::memcmp(&a[i], &b[i], sizeof(T)) == 0)
        continue;
      std::cerr << "preparation n=" << n << " start=" << start
                << " rotary=" << rotary << " replay=" << replay
                << " index=" << i << " expected=" << std::hexfloat
                << static_cast<float>(a[i])
                << " actual=" << static_cast<float>(b[i]) << std::defaultfloat
                << '\n';
      throw std::runtime_error(std::string("attention preparation changed ") +
                               name);
    }
  };
  for (std::uint32_t replay = 0; replay < 2; ++replay) {
    const std::vector<std::uint32_t> position{start + replay};
    Upload(&pos, position);
    q::UnpackQGate(packed.get(), stride, q_ref.get(), gate_ref.get(), k.get(),
                   v.get(), n, kHeads, kDim, kKvWidth, nullptr);
    q::RmsNormRows(q_ref.get(), q_gamma.get(), q_ref.get(), n * kHeads, kDim, 1,
                   1e-6F, nullptr);
    q::RmsNormRows(k.get(), k_gamma.get(), k.get(), n * kKvHeads, kDim, 1,
                   1e-6F, nullptr);
    q::Rope(q_ref.get(), n, kHeads, kDim, rotary, pos.get(), 1e7F, nullptr);
    q::Rope(k.get(), n, kKvHeads, kDim, rotary, pos.get(), 1e7F, nullptr);
    q::StoreKv(k.get(), k_ref.get(), n, kKvWidth, pos.get(), nullptr);
    q::StoreKv(v.get(), v_ref.get(), n, kKvWidth, pos.get(), nullptr);
    CheckHip(hipDeviceSynchronize(), "preparation reference ready");
    if (replay == 0) {
      CheckHip(
          hipStreamBeginCapture(capture.stream, hipStreamCaptureModeGlobal),
          "preparation capture");
      const bool supported = q::PrepareAttention(
          packed.get(), stride, q_gamma.get(), k_gamma.get(), q_out.get(),
          gate_out.get(), k_out.get(), v_out.get(), n, kHeads, kKvHeads, kDim,
          rotary, pos.get(), 1e7F, 1e-6F, capture.stream);
      CheckHip(hipStreamEndCapture(capture.stream, &capture.graph),
               "preparation capture end");
      if (!supported)
        throw std::runtime_error("attention preparation rejected geometry");
      CheckHip(hipGraphInstantiate(&capture.exec, capture.graph, nullptr,
                                   nullptr, 0),
               "preparation instantiate");
    }
    CheckHip(hipGraphLaunch(capture.exec, capture.stream),
             "preparation replay");
    CheckHip(hipStreamSynchronize(capture.stream), "preparation ready");
    same(q_ref.get(), q_out.get(), count, "queries", replay);
    same(gate_ref.get(), gate_out.get(), count, "gates", replay);
    // Include the neighboring cache rows and replay at a new device position.
    const std::size_t first = start == 0 ? 0 : start - 1;
    const std::size_t window = (cache_rows - first) * kKvWidth;
    same(k_ref.get() + first * kKvWidth, k_out.get() + first * kKvWidth, window,
         "key cache", replay);
    same(v_ref.get() + first * kKvWidth, v_out.get() + first * kKvWidth, window,
         "value cache", replay);
  }
  if (n >= 32) {
    // Each small prefill piece must match the full preparation, including
    // normalization rounding and absolute rotary positions.
    for (const std::uint32_t chunk : {1U, 8U, 9U, 16U, 32U}) {
      for (std::uint32_t off = 0; off < n; off += chunk) {
        Upload(&pos, std::vector<std::uint32_t>{start + 1 + off});
        if (!q::PrepareAttention(
                packed.get() + std::size_t(off) * stride, stride, q_gamma.get(),
                k_gamma.get(), q_out.get() + std::size_t(off) * kQWidth,
                gate_out.get() + std::size_t(off) * kQWidth, k_out.get(),
                v_out.get(), std::min(chunk, n - off), kHeads, kKvHeads, kDim,
                rotary, pos.get(), 1e7F, 1e-6F, nullptr, nullptr, true))
          throw std::runtime_error("prefill preparation rejected a short tail");
      }
      CheckHip(hipDeviceSynchronize(), "short prefill preparation");
      same(q_ref.get(), q_out.get(), count, "prefill queries", chunk);
      same(gate_ref.get(), gate_out.get(), count, "prefill gates", chunk);
      same(k_ref.get(), k_out.get(), cache_rows * kKvWidth, "prefill keys",
           chunk);
      same(v_ref.get(), v_out.get(), cache_rows * kKvWidth, "prefill values",
           chunk);
    }
  }
  std::cout << "attention preparation n=" << n << " start=" << start
            << " rotary=" << rotary << ": exact, including graph positions\n";
}

/// Runs the per-token reference and the fused WMMA route over the same
/// cache and reports the worst absolute error of the gated context.
double Compare(std::uint32_t n_tokens, std::uint32_t start_pos, bool masked,
               std::uint32_t seed, bool sparse = false, bool bounded = false) {
  const std::uint32_t n_kv = start_pos + n_tokens;
  const std::uint32_t max_blocks = (n_kv + kRatio - 1) / kRatio;
  const std::uint32_t mask_words = (max_blocks + 31) / 32;
  const std::size_t q_count = static_cast<std::size_t>(n_tokens) * kQWidth;
  const std::size_t kv_count = static_cast<std::size_t>(n_kv) * kKvWidth;

  const auto qv = MakeValues(q_count, seed, 4.0F);
  const auto gate = MakeValues(q_count, seed ^ 0x5555U, 3.0F);
  const auto kf = MakeValues(kv_count, seed ^ 0xAAAAU, 1.0F);
  const auto vf = MakeValues(kv_count, seed ^ 0x3333U, 1.0F);
  std::vector<__half> kh(kv_count);
  std::vector<__half> vh(kv_count);
  for (std::size_t i = 0; i < kv_count; ++i) {
    kh[i] = __float2half(kf[i]);
    vh[i] = __float2half(vf[i]);
  }
  std::vector<std::uint32_t> mask(static_cast<std::size_t>(n_tokens) *
                                  mask_words);
  std::uint32_t state = seed ^ 0x77777777U;
  for (std::uint32_t& word : mask) {
    word = NextRandom(&state) & NextRandom(&state);
    if (sparse) {
      // About one block in sixteen: most 16-key tiles are unselected for a
      // whole query block, which is what the tile skipping is for.
      word &= NextRandom(&state) & NextRandom(&state);
    }
  }
  if (bounded) {
    std::fill(mask.begin(), mask.end(), 0);
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const auto complete = (start_pos + t + 1) / kRatio;
      std::uint32_t selected = 0;
      while (selected < std::min(512U, complete)) {
        const auto block = NextRandom(&state) % complete;
        auto& word =
            mask[static_cast<std::size_t>(t) * mask_words + block / 32];
        const auto bit = 1U << (block % 32);
        if ((word & bit) == 0) {
          word |= bit;
          ++selected;
        }
      }
    }
  }

  HipBuffer<float> d_q(q_count);
  HipBuffer<float> d_gate(q_count);
  HipBuffer<__half> d_k(kv_count);
  HipBuffer<__half> d_v(kv_count);
  HipBuffer<std::uint32_t> d_mask(mask.size());
  HipBuffer<std::uint32_t> d_pos(1);
  HipBuffer<float> d_ref(q_count);
  HipBuffer<float> d_wmma(q_count);
  Upload(&d_q, qv);
  Upload(&d_gate, gate);
  Upload(&d_k, kh);
  Upload(&d_v, vh);
  Upload(&d_mask, mask);
  Upload(&d_pos, std::vector<std::uint32_t>{start_pos});
  const std::uint32_t* mask_ptr = masked ? d_mask.get() : nullptr;

  q::Attention(d_q.get(), d_k.get(), d_v.get(), mask_ptr, mask_words,
               d_ref.get(), nullptr, 1, n_tokens, d_pos.get(), kHeads, kKvHeads,
               kDim, kRatio, nullptr);
  q::SigmoidMul(d_ref.get(), d_gate.get(), q_count, nullptr);
  // The split-key form of the reference must match its single-block form
  // (log-sum-exp merge), the way decode runs it.
  {
    constexpr std::uint32_t kSplits = 8;
    HipBuffer<float> d_split(q_count);
    HipBuffer<float> d_partials(static_cast<std::size_t>(n_tokens) * kHeads *
                                kSplits * (kDim + 2));
    q::Attention(d_q.get(), d_k.get(), d_v.get(), mask_ptr, mask_words,
                 d_split.get(), d_partials.get(), kSplits, n_tokens,
                 d_pos.get(), kHeads, kKvHeads, kDim, kRatio, nullptr);
    q::SigmoidMul(d_split.get(), d_gate.get(), q_count, nullptr);
    CheckHip(hipDeviceSynchronize(), "split attention");
    const auto a = Download(&d_ref, q_count);
    const auto b = Download(&d_split, q_count);
    double worst = 0.0;
    for (std::size_t i = 0; i < q_count; ++i) {
      worst = std::max(worst, std::abs(static_cast<double>(a[i] - b[i])));
    }
    std::cout << "  split-key reference worst absolute error " << worst << '\n';
    if (worst > 1e-4) {
      throw std::runtime_error("split-key attention disagrees");
    }
  }
  if (!q::WmmaCausalAttention(d_q.get(), d_gate.get(), d_k.get(), d_v.get(),
                              mask_ptr, mask_words, d_wmma.get(), n_tokens,
                              start_pos, kHeads, kKvHeads, kDim, kRatio,
                              nullptr)) {
    throw std::runtime_error("WMMA attention rejected the model geometry");
  }
  CheckHip(hipDeviceSynchronize(), "attention synchronization");

  const auto ref = Download(&d_ref, q_count);
  const auto out = Download(&d_wmma, q_count);
  if (!q::WmmaCausalAttention(d_q.get(), d_gate.get(), d_k.get(), d_v.get(),
                              mask_ptr, mask_words, d_wmma.get(), n_tokens,
                              start_pos, kHeads, kKvHeads, kDim, kRatio,
                              nullptr)) {
    throw std::runtime_error("WMMA attention replay rejected the geometry");
  }
  const auto replay = Download(&d_wmma, q_count);
  if (std::memcmp(out.data(), replay.data(), q_count * sizeof(float)) != 0) {
    throw std::runtime_error("WMMA attention replay changed the output");
  }
  {
    constexpr float poison = -12345.0F;
    Upload(&d_wmma, std::vector<float>(q_count, poison));
    if (!q::WmmaCausalAttention(d_q.get(), d_gate.get(), d_k.get(), d_v.get(),
                                mask_ptr, mask_words, d_wmma.get(), n_tokens,
                                start_pos, kHeads, kKvHeads, kDim, kRatio,
                                nullptr, true)) {
      throw std::runtime_error("last-tile attention rejected the geometry");
    }
    const auto tail = Download(&d_wmma, q_count);
    const std::uint32_t tile_rows = mask_ptr ? 4 : 16;
    const std::size_t begin =
        static_cast<std::size_t>((n_tokens - 1) / tile_rows * tile_rows) *
        kQWidth;
    for (std::size_t i = 0; i < q_count; ++i) {
      if (i < begin ? tail[i] != poison
                    : std::memcmp(&out[i], &tail[i], sizeof(float)) != 0) {
        throw std::runtime_error("last-tile attention changed a row or guard");
      }
    }
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < q_count; ++i) {
    if (!std::isfinite(out[i])) {
      throw std::runtime_error("WMMA attention output is not finite");
    }
    worst = std::max(worst, std::abs(static_cast<double>(ref[i] - out[i])));
  }
  if (worst > 1e-2) {
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      for (std::uint32_t h = 0; h < kHeads; ++h) {
        double w = 0.0;
        for (std::uint32_t d = 0; d < kDim; ++d) {
          const std::size_t i =
              (static_cast<std::size_t>(t) * kHeads + h) * kDim + d;
          w = std::max(w, std::abs(static_cast<double>(ref[i] - out[i])));
        }
        if (w > 1e-2) {
          std::cout << "  query " << t << " head " << h << " err " << w << '\n';
        }
      }
    }
  }
  return worst;
}

void CheckChunks(std::uint32_t n, std::uint32_t split) {
  const std::size_t count = std::size_t(n) * kQWidth;
  HipBuffer<float> queries(count), gates(count), full(count), chunked(count);
  HipBuffer<__half> keys(std::size_t(n) * kKvWidth),
      values(std::size_t(n) * kKvWidth);
  Upload(&queries, MakeValues(count, 412, 4.0F));
  Upload(&gates, MakeValues(count, 721, 3.0F));
  for (auto* destination : {&keys, &values}) {
    const auto f = MakeValues(std::size_t(n) * kKvWidth,
                              destination == &keys ? 891 : 347, 1.0F);
    std::vector<__half> half(f.size());
    std::transform(f.begin(), f.end(), half.begin(),
                   [](float x) { return __float2half(x); });
    Upload(destination, half);
  }
  const auto run = [&](std::uint32_t start, std::uint32_t rows, float* out) {
    const auto offset = std::size_t(start) * kQWidth;
    if (!q::WmmaCausalAttention(queries.get() + offset, gates.get() + offset,
                                keys.get(), values.get(), nullptr, 0,
                                out + offset, rows, start, kHeads, kKvHeads,
                                kDim, kRatio, nullptr)) {
      throw std::runtime_error("chunk attention rejected geometry");
    }
  };
  run(0, n, full.get());
  run(0, split, chunked.get());
  run(split, n - split, chunked.get());
  const auto a = Download(&full, count);
  const auto b = Download(&chunked, count);
  std::size_t differences = 0, first = count;
  float max_error = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (a[i] != b[i]) {
      ++differences;
      first = std::min(first, i);
      max_error = std::max(max_error, std::abs(a[i] - b[i]));
    }
  }
  std::cout << "chunk attention n=" << n << " split=" << split
            << " differing=" << differences << " max=" << max_error
            << " first_row=" << first / kQWidth << '\n';
  if (differences != 0)
    throw std::runtime_error("attention depends on prefill chunk boundary");
}

// ---- TurboQuant KV caches: host FP64 model of the kernels.hpp contract ----

using gufo::models::qwen38_flash_next::KvCacheFormat;

double Normal(std::uint32_t* state) {
  const double u1 =
      (static_cast<double>(NextRandom(state)) + 1.0) / 4294967297.0;
  const double u2 = static_cast<double>(NextRandom(state)) / 4294967296.0;
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

// Heavy-tailed channel noise with Qwen-like outlier channels. Keys are
// RMS-normalized with an outlier-heavy gamma and rotated (NEOX, 64 dims) at
// their token position; values keep their raw scale.
std::vector<float> QwenLikeRows(std::size_t tokens, std::uint32_t heads,
                                std::uint32_t seed, bool key,
                                std::uint32_t first_position = 0) {
  std::uint32_t state = seed | 1U;
  std::vector<double> channel(kDim), gamma(kDim);
  for (std::uint32_t i = 0; i < kDim; ++i) {
    channel[i] = std::exp((key ? 0.6 : 0.8) * Normal(&state));
    gamma[i] = std::exp(0.35 * Normal(&state));
  }
  for (int j = 0; j < 6; ++j)
    channel[NextRandom(&state) % kDim] *= key ? 12.0 : 10.0;
  for (int j = 0; j < 4; ++j)
    gamma[NextRandom(&state) % kDim] *= 4.0;
  std::vector<float> out(tokens * heads * kDim);
  std::vector<double> row(kDim);
  for (std::size_t r = 0; r < tokens * heads; ++r) {
    for (std::uint32_t i = 0; i < kDim; ++i) {
      // Normal times a lognormal magnitude: heavier tails than Gaussian.
      const double u1 =
          (static_cast<double>(NextRandom(&state)) + 1.0) / 4294967297.0;
      const double u2 = static_cast<double>(NextRandom(&state)) / 4294967296.0;
      const double radius = std::sqrt(-2.0 * std::log(u1));
      const double angle = 6.283185307179586 * u2;
      row[i] = radius * std::cos(angle) *
               std::exp(0.4 * radius * std::sin(angle)) * channel[i];
    }
    if (key) {
      double ss = 0.0;
      for (double x : row)
        ss += x * x;
      const double scale = 1.0 / std::sqrt(ss / kDim + 1e-6);
      for (std::uint32_t i = 0; i < kDim; ++i)
        row[i] *= scale * gamma[i];
      const double position = first_position + static_cast<double>(r / heads);
      for (std::uint32_t i = 0; i < 32; ++i) {
        const double angle = position * std::pow(1e7, -2.0 * i / 64.0);
        const double a = row[i], b = row[i + 32];
        row[i] = a * std::cos(angle) - b * std::sin(angle);
        row[i + 32] = a * std::sin(angle) + b * std::cos(angle);
      }
    }
    for (std::uint32_t i = 0; i < kDim; ++i)
      out[r * kDim + i] = static_cast<float>(row[i]);
  }
  return out;
}

bool Negated(std::uint32_t e) {
  return ((q::kTqSigns[e / 32] >> (e % 32)) & 1U) != 0;
}
std::uint32_t StoragePosition(std::uint32_t e) {
  return (e % 32) * 8 + e / 32;
}

void HostHadamard(std::vector<double>& x) {
  for (std::uint32_t h = 1; h < kDim; h <<= 1) {
    for (std::uint32_t i = 0; i < kDim; i += 2 * h) {
      for (std::uint32_t j = i; j < i + h; ++j) {
        const double a = x[j], b = x[j + h];
        x[j] = a + b;
        x[j + h] = a - b;
      }
    }
  }
  for (double& v : x)
    v /= 16.0;
}

/// R x by storage position.
std::vector<double> HostRotate(const float* x) {
  std::vector<double> y(kDim), out(kDim);
  for (std::uint32_t e = 0; e < kDim; ++e)
    y[e] = Negated(e) ? -static_cast<double>(x[e]) : x[e];
  HostHadamard(y);
  for (std::uint32_t e = 0; e < kDim; ++e)
    out[StoragePosition(e)] = y[e];
  return out;
}

/// R^T of a row given by storage position.
std::vector<double> HostUnrotate(const std::vector<double>& positions) {
  std::vector<double> y(kDim);
  for (std::uint32_t e = 0; e < kDim; ++e)
    y[e] = positions[StoragePosition(e)];
  HostHadamard(y);
  for (std::uint32_t e = 0; e < kDim; ++e)
    y[e] = Negated(e) ? -y[e] : y[e];
  return y;
}

std::uint32_t HostCode(KvCacheFormat format, const std::uint8_t* row,
                       std::uint32_t p) {
  const std::uint32_t lane = p / 8, b = p % 4, upper = (p % 8) / 4;
  if (format == KvCacheFormat::kTq4)
    return (row[lane * 4 + b] >> (4 * upper)) & 15U;
  const std::uint32_t low = row[lane * 2] | row[lane * 2 + 1] << 8;
  return ((low >> (4 * b + 2 * upper)) & 3U) |
         ((row[64 + lane] >> (2 * b + upper)) & 1U) << 2;
}

double HostCentroid(KvCacheFormat format, std::uint32_t code) {
  const bool k4 = format == KvCacheFormat::kTq4;
  const std::uint32_t half = k4 ? 8 : 4;
  const float* table = k4 ? q::kTq4Centroids : q::kTq3Centroids;
  return code >= half ? table[code - half] : -table[half - 1 - code];
}

/// FP64 nearest-centroid codes of a rotated row (the store contract).
std::vector<std::uint32_t> HostCodes(KvCacheFormat format,
                                     const std::vector<double>& y) {
  const bool k4 = format == KvCacheFormat::kTq4;
  double ss = 0.0;
  for (double v : y)
    ss += v * v;
  const double inv =
      ss > 0.0 ? (k4 ? q::kTq4Unit : q::kTq3Unit) / std::sqrt(ss / kDim) : 0.0;
  std::vector<std::uint32_t> codes(kDim);
  const std::uint32_t count = k4 ? 16 : 8;
  for (std::uint32_t p = 0; p < kDim; ++p) {
    std::uint32_t best = 0;
    for (std::uint32_t c = 1; c < count; ++c) {
      if (std::abs(y[p] * inv - HostCentroid(format, c)) <
          std::abs(y[p] * inv - HostCentroid(format, best)))
        best = c;
    }
    codes[p] = best;
  }
  return codes;
}

struct TqRows {
  std::vector<std::uint8_t> codes;
  std::vector<__half> scales;
};

/// Natural-order reconstruction of cache row r.
std::vector<double> HostDecode(KvCacheFormat format, const TqRows& rows,
                               std::size_t r) {
  const std::uint8_t* row = rows.codes.data() + r * q::KvCodeBytes(format);
  const double scale = __half2float(rows.scales[r]);
  std::vector<double> positions(kDim);
  for (std::uint32_t p = 0; p < kDim; ++p)
    positions[p] = scale * HostCentroid(format, HostCode(format, row, p));
  return HostUnrotate(positions);
}

/// Device TurboQuant K/V cache of `tokens` positions.
struct TqCache {
  TqCache(KvCacheFormat format, std::size_t tokens)
      : format(format),
        rows(tokens * kKvHeads),
        k(rows * q::KvCodeBytes(format)),
        v(rows * q::KvCodeBytes(format)),
        ks(rows),
        vs(rows) {}
  q::KvView View() {
    return q::KvView{format, k.get(), v.get(), ks.get(), vs.get()};
  }
  TqRows Download(bool value) {
    TqRows out{std::vector<std::uint8_t>(rows * q::KvCodeBytes(format)),
               std::vector<__half>(rows)};
    CheckHip(hipMemcpy(out.codes.data(), (value ? v : k).get(),
                       out.codes.size(), hipMemcpyDeviceToHost),
             "download codes");
    CheckHip(hipMemcpy(out.scales.data(), (value ? vs : ks).get(),
                       rows * sizeof(__half), hipMemcpyDeviceToHost),
             "download scales");
    return out;
  }
  KvCacheFormat format;
  std::size_t rows;
  HipBuffer<std::uint8_t> k, v;
  HipBuffer<__half> ks, vs;
};

const char* FormatName(KvCacheFormat format) {
  return format == KvCacheFormat::kTq4 ? "tq4" : "tq3";
}

/// QuantizeKv against the FP64 contract: codes, scales, reconstruction.
void CheckTqStore(KvCacheFormat format) {
  constexpr std::size_t kTokens = 3001;
  constexpr std::uint32_t kStart = 5;
  const auto kf = QwenLikeRows(kTokens, kKvHeads, 101, true);
  const auto vf = QwenLikeRows(kTokens, kKvHeads, 202, false);
  HipBuffer<float> dk(kf.size()), dv(vf.size());
  HipBuffer<std::uint32_t> pos(1);
  Upload(&dk, kf);
  Upload(&dv, vf);
  Upload(&pos, std::vector<std::uint32_t>{kStart});
  TqCache cache(format, kStart + kTokens);
  q::QuantizeKv(dk.get(), dv.get(), cache.View(), kTokens, kKvHeads, pos.get(),
                nullptr);
  CheckHip(hipDeviceSynchronize(), "quantize");
  std::size_t code_mismatch = 0;
  double worst_scale = 0.0, err_sum = 0.0, err_max = 0.0;
  for (bool value : {false, true}) {
    const TqRows rows = cache.Download(value);
    const auto& src = value ? vf : kf;
    for (std::size_t r = 0; r < kTokens * kKvHeads; ++r) {
      const std::size_t slot = kStart * kKvHeads + r;
      const auto y = HostRotate(src.data() + r * kDim);
      const auto codes = HostCodes(format, y);
      double yc = 0.0, cc = 0.0;
      const std::uint8_t* row =
          rows.codes.data() + slot * q::KvCodeBytes(format);
      for (std::uint32_t p = 0; p < kDim; ++p) {
        const std::uint32_t code = HostCode(format, row, p);
        code_mismatch += code != codes[p] ? 1 : 0;
        const double c = HostCentroid(format, code);
        yc += y[p] * c;
        cc += c * c;
      }
      const double scale = yc / cc;
      worst_scale = std::max(
          worst_scale,
          std::abs(__half2float(rows.scales[slot]) - scale) / std::abs(scale));
      const auto x = HostDecode(format, rows, slot);
      double e2 = 0.0, x2 = 0.0;
      for (std::uint32_t i = 0; i < kDim; ++i) {
        const double d = x[i] - src[r * kDim + i];
        e2 += d * d;
        x2 += double(src[r * kDim + i]) * src[r * kDim + i];
      }
      const double rel = std::sqrt(e2 / x2);
      err_sum += rel;
      err_max = std::max(err_max, rel);
    }
  }
  const double rows_checked = 2.0 * kTokens * kKvHeads;
  std::cout << FormatName(format) << " store: code mismatches " << code_mismatch
            << " of " << rows_checked * kDim << ", worst scale rel "
            << worst_scale << ", row rel L2 mean " << err_sum / rows_checked
            << " max " << err_max << '\n';
  const double bound = format == KvCacheFormat::kTq4 ? 0.2 : 0.35;
  if (code_mismatch > rows_checked * kDim * 1e-4 || worst_scale > 1e-3 ||
      err_max > bound || !(err_sum / rows_checked > 0.0))
    throw std::runtime_error("TurboQuant store disagrees with its contract");
}

/// FP64 gated attention of query rows [first, first + count) over host
/// K/V rows ([pos][kv_heads][256]), following Attention's mask contract.
std::vector<double> HostAttention(const std::vector<float>& queries,
                                  const std::vector<float>& gates,
                                  const std::vector<double>& keys,
                                  const std::vector<double>& values,
                                  const std::vector<std::uint32_t>* mask,
                                  std::uint32_t mask_words, std::uint32_t start,
                                  std::uint32_t first, std::uint32_t count) {
  std::vector<double> out(std::size_t(count) * kQWidth);
  std::vector<double> scores;
  for (std::uint32_t t = first; t < first + count; ++t) {
    const std::uint32_t position = start + t;
    const std::uint32_t tail = (position + 1) / kRatio * kRatio;
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      const std::uint32_t kvh = h / (kHeads / kKvHeads);
      const float* qrow = queries.data() + (std::size_t(t) * kHeads + h) * kDim;
      scores.assign(position + 1, -INFINITY);
      double top = -INFINITY;
      for (std::uint32_t j = 0; j <= position; ++j) {
        if (mask != nullptr && j < tail) {
          const std::uint32_t b = j / kRatio;
          if (((*mask)[std::size_t(t) * mask_words + b / 32] >> (b % 32) &
               1U) == 0)
            continue;
        }
        const double* krow =
            keys.data() + (std::size_t(j) * kKvHeads + kvh) * kDim;
        double dot = 0.0;
        for (std::uint32_t i = 0; i < kDim; ++i)
          dot += double(qrow[i]) * krow[i];
        scores[j] = dot / 16.0;
        top = std::max(top, scores[j]);
      }
      double sum = 0.0;
      std::vector<double> acc(kDim, 0.0);
      for (std::uint32_t j = 0; j <= position; ++j) {
        if (scores[j] == -INFINITY)
          continue;
        const double w = std::exp(scores[j] - top);
        sum += w;
        const double* vrow =
            values.data() + (std::size_t(j) * kKvHeads + kvh) * kDim;
        for (std::uint32_t i = 0; i < kDim; ++i)
          acc[i] += w * vrow[i];
      }
      for (std::uint32_t i = 0; i < kDim; ++i) {
        const std::size_t o = (std::size_t(t) * kHeads + h) * kDim + i;
        const double g = 1.0 / (1.0 + std::exp(-double(gates[o])));
        out[(std::size_t(t - first) * kHeads + h) * kDim + i] =
            acc[i] / sum * g;
      }
    }
  }
  return out;
}

/// TurboQuant decode (per-token, split and unsplit) and WMMA attention
/// against an FP64 reference over the dequantized cache, plus the error
/// against unquantized K/V (the quality cost of the format).
void CheckTqAttention(KvCacheFormat format, std::uint32_t n_tokens,
                      std::uint32_t start, bool masked, bool wmma,
                      std::uint32_t seed) {
  const std::uint32_t n_kv = start + n_tokens;
  const std::uint32_t max_blocks = (n_kv + kRatio - 1) / kRatio;
  const std::uint32_t mask_words = (max_blocks + 31) / 32;
  const std::size_t q_count = std::size_t(n_tokens) * kQWidth;
  const auto kf = QwenLikeRows(n_kv, kKvHeads, seed, true);
  const auto vf = QwenLikeRows(n_kv, kKvHeads, seed ^ 0x55U, false);
  auto qf =
      QwenLikeRows(std::size_t(n_tokens), kHeads, seed ^ 0xAAU, true, start);
  // Retrieval-like heads: a third of the queries lean towards one key.
  std::uint32_t state = seed ^ 0x1234U;
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    for (std::uint32_t h = 0; h < kHeads; h += 3) {
      const std::uint32_t j = NextRandom(&state) % (start + t + 1);
      const float* krow =
          kf.data() + (std::size_t(j) * kKvHeads + h / 12) * kDim;
      float* qrow = qf.data() + (std::size_t(t) * kHeads + h) * kDim;
      for (std::uint32_t i = 0; i < kDim; ++i)
        qrow[i] += 1.5F * krow[i];
    }
  }
  const auto gate = MakeValues(q_count, seed ^ 0x777U, 3.0F);
  std::vector<std::uint32_t> mask(std::size_t(n_tokens) * mask_words, 0U);
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const std::uint32_t complete = (start + t + 1) / kRatio;
    std::uint32_t selected = 0;
    while (selected < std::min(512U, complete)) {
      const std::uint32_t block = NextRandom(&state) % complete;
      auto& word = mask[std::size_t(t) * mask_words + block / 32];
      if ((word & (1U << (block % 32))) == 0) {
        word |= 1U << (block % 32);
        ++selected;
      }
    }
  }
  HipBuffer<float> dk(kf.size()), dv(vf.size()), dq(q_count), dgate(q_count),
      drot(q_count), dout(q_count);
  HipBuffer<std::uint32_t> dmask(mask.size()), dpos(1), dzero(1);
  Upload(&dk, kf);
  Upload(&dv, vf);
  Upload(&dq, qf);
  Upload(&dgate, gate);
  Upload(&dmask, mask);
  Upload(&dpos, std::vector<std::uint32_t>{start});
  Upload(&dzero, std::vector<std::uint32_t>{0U});
  TqCache cache(format, n_kv);
  q::QuantizeKv(dk.get(), dv.get(), cache.View(), n_kv, kKvHeads, dzero.get(),
                nullptr);
  CheckHip(hipDeviceSynchronize(), "attention cache");
  const TqRows k_rows = cache.Download(false), v_rows = cache.Download(true);
  std::vector<double> kq(kf.size()), vq(vf.size()), kx(kf.begin(), kf.end()),
      vx(vf.begin(), vf.end());
  for (std::size_t r = 0; r < std::size_t(n_kv) * kKvHeads; ++r) {
    const auto a = HostDecode(format, k_rows, r);
    const auto b = HostDecode(format, v_rows, r);
    std::copy(a.begin(), a.end(), kq.begin() + r * kDim);
    std::copy(b.begin(), b.end(), vq.begin() + r * kDim);
  }
  const std::vector<std::uint32_t>* host_mask = masked ? &mask : nullptr;
  const std::uint32_t* mask_ptr = masked ? dmask.get() : nullptr;
  // Bound the FP64 reference to the final rows of wide cases.
  const std::uint32_t first = n_tokens > 9 ? n_tokens - 9 : 0;
  const std::uint32_t count = n_tokens - first;
  const auto ref = HostAttention(qf, gate, kq, vq, host_mask, mask_words, start,
                                 first, count);
  const auto exact = HostAttention(qf, gate, kx, vx, host_mask, mask_words,
                                   start, first, count);
  // Worst error relative to the largest reference output (outlier value
  // channels reach tens), and the format's cost against exact K/V.
  const auto compare = [&](const char* route) {
    const auto out = Download(&dout, q_count);
    double worst = 0.0, top = 1.0, e2 = 0.0, r2 = 0.0, q2 = 0.0;
    for (std::size_t i = 0; i < std::size_t(count) * kQWidth; ++i) {
      const double o = out[std::size_t(first) * kQWidth + i];
      if (!std::isfinite(o))
        throw std::runtime_error(std::string(route) + " output is not finite");
      worst = std::max(worst, std::abs(o - ref[i]));
      top = std::max(top, std::abs(ref[i]));
      e2 += (o - exact[i]) * (o - exact[i]);
      q2 += (ref[i] - exact[i]) * (ref[i] - exact[i]);
      r2 += exact[i] * exact[i];
    }
    std::cout << "  " << FormatName(format) << ' ' << route << " n=" << n_tokens
              << " start=" << start << (masked ? " sparse" : " dense")
              << ": vs dequantized FP64 max abs " << worst << " (of " << top
              << "), vs unquantized rel L2 " << std::sqrt(e2 / r2)
              << " (FP64 format error " << std::sqrt(q2 / r2) << ")\n";
    return worst / top;
  };
  const auto rotated_queries = [&]() {
    CheckHip(hipMemcpy(drot.get(), dq.get(), q_count * sizeof(float),
                       hipMemcpyDeviceToDevice),
             "copy queries");
    q::RotateHeads(drot.get(), n_tokens * kHeads, nullptr);
  };
  if (!wmma) {
    for (const std::uint32_t splits : {1U, 8U}) {
      HipBuffer<float> partials(std::size_t(n_tokens) * kHeads * splits *
                                (kDim + 2));
      rotated_queries();
      q::Attention(drot.get(), cache.View(), mask_ptr, mask_words, dout.get(),
                   splits > 1 ? partials.get() : nullptr, splits, n_tokens,
                   dpos.get(), kHeads, kKvHeads, kDim, kRatio, nullptr);
      q::UnrotateHeads(dout.get(), dgate.get(), n_tokens * kHeads, nullptr);
      CheckHip(hipDeviceSynchronize(), "TurboQuant decode attention");
      if (compare(splits > 1 ? "decode split" : "decode") > 1e-4)
        throw std::runtime_error("TurboQuant decode attention disagrees");
    }
    return;
  }
  rotated_queries();
  if (!q::WmmaCausalAttention(drot.get(), nullptr, cache.View(), mask_ptr,
                              mask_words, dout.get(), n_tokens, start, kHeads,
                              kKvHeads, kDim, kRatio, nullptr))
    throw std::runtime_error("TurboQuant WMMA rejected the geometry");
  q::UnrotateHeads(dout.get(), dgate.get(), n_tokens * kHeads, nullptr);
  CheckHip(hipDeviceSynchronize(), "TurboQuant WMMA attention");
  // F16 queries and probabilities, as for the F16 cache (2e-2 there).
  if (compare("wmma") > 2e-2)
    throw std::runtime_error("TurboQuant WMMA attention disagrees");
}

/// PrepareAttention must store exactly what the separate normalization,
/// rotary and QuantizeKv kernels store.
void CheckTqPreparation(KvCacheFormat format, std::uint32_t n,
                        std::uint32_t start, bool prefill) {
  constexpr std::uint32_t stride = 2 * (kQWidth + kKvWidth);
  HipBuffer<float> packed(std::size_t(n) * stride);
  Upload(&packed, MakeValues(std::size_t(n) * stride, 29, 4.0F));
  auto g = MakeValues(2 * kDim, 57, 0.5F);
  for (auto& x : g)
    x += 1.0F;
  HipBuffer<float> gamma(2 * kDim);
  Upload(&gamma, g);
  const std::size_t count = std::size_t(n) * kQWidth;
  HipBuffer<float> q_ref(count), gate_ref(count), q_out(count), gate_out(count);
  HipBuffer<float> k(std::size_t(n) * kKvWidth), v(std::size_t(n) * kKvWidth);
  HipBuffer<std::uint32_t> pos(1);
  Upload(&pos, std::vector<std::uint32_t>{start});
  TqCache ref(format, start + n), out(format, start + n);
  q::UnpackQGate(packed.get(), stride, q_ref.get(), gate_ref.get(), k.get(),
                 v.get(), n, kHeads, kDim, kKvWidth, nullptr);
  q::RmsNormRows(k.get(), gamma.get() + kDim, k.get(), n * kKvHeads, kDim, 1,
                 1e-6F, nullptr);
  q::Rope(k.get(), n, kKvHeads, kDim, 64, pos.get(), 1e7F, nullptr);
  q::QuantizeKv(k.get(), v.get(), ref.View(), n, kKvHeads, pos.get(), nullptr);
  if (!q::PrepareAttention(packed.get(), stride, gamma.get(),
                           gamma.get() + kDim, q_out.get(), gate_out.get(),
                           out.View(), n, kHeads, kKvHeads, kDim, 64, pos.get(),
                           1e7F, 1e-6F, nullptr, nullptr, prefill))
    throw std::runtime_error("TurboQuant preparation rejected geometry");
  CheckHip(hipDeviceSynchronize(), "TurboQuant preparation");
  for (bool value : {false, true}) {
    const TqRows a = ref.Download(value), b = out.Download(value);
    const std::size_t first = std::size_t(start) * kKvHeads;
    if (!std::equal(a.codes.begin() + first * q::KvCodeBytes(format),
                    a.codes.end(),
                    b.codes.begin() + first * q::KvCodeBytes(format)) ||
        std::memcmp(a.scales.data() + first, b.scales.data() + first,
                    (a.scales.size() - first) * sizeof(__half)) != 0)
      throw std::runtime_error(std::string("TurboQuant preparation changed ") +
                               (value ? "values" : "keys"));
  }
  std::cout << FormatName(format) << " preparation n=" << n
            << " start=" << start << (prefill ? " prefill" : "")
            << ": exact against separate kernels\n";
}

/// One session's device state for the multi-session check.
struct BatchSession {
  BatchSession(KvCacheFormat format, std::uint32_t position, std::uint32_t n,
               std::uint32_t seed)
      : format(format),
        position(position),
        n(n),
        context(position + n),
        // An unused one-row TurboQuant cache for f16 sessions.
        cache(format == KvCacheFormat::kF16 ? KvCacheFormat::kTq4 : format,
              format == KvCacheFormat::kF16 ? 1 : context),
        f16k(format == KvCacheFormat::kF16 ? std::size_t(context) * kKvWidth
                                           : 1),
        f16v(format == KvCacheFormat::kF16 ? std::size_t(context) * kKvWidth
                                           : 1),
        ring(std::size_t(kRing) * 128),
        blocks(std::size_t(context / kRatio + 1) * 128),
        control(2) {
    Upload(&ring, MakeValues(std::size_t(kRing) * 128, seed, 1.0F));
    std::vector<__half> b(std::size_t(context / kRatio + 1) * 128);
    const auto bf = MakeValues(b.size(), seed ^ 0x99U, 1.0F);
    for (std::size_t i = 0; i < b.size(); ++i)
      b[i] = __float2half(bf[i]);
    Upload(&blocks, b);
    Upload(&control, std::vector<std::uint32_t>{position, position / kRatio});
    // Earlier cache rows: quantize (or store) random rows once.
    HipBuffer<float> rows(std::size_t(context) * kKvWidth);
    Upload(&rows, MakeValues(std::size_t(context) * kKvWidth, seed ^ 7U, 1.0F));
    HipBuffer<std::uint32_t> zero(1);
    if (format == KvCacheFormat::kF16) {
      q::StoreKv(rows.get(), f16k.get(), context, kKvWidth, zero.get(),
                 nullptr);
      q::StoreKv(rows.get(), f16v.get(), context, kKvWidth, zero.get(),
                 nullptr);
    } else {
      q::QuantizeKv(rows.get(), rows.get(), cache.View(), context, kKvHeads,
                    zero.get(), nullptr);
    }
  }
  q::KvView View() {
    return format == KvCacheFormat::kF16
               ? q::KvView{format, f16k.get(), f16v.get()}
               : cache.View();
  }
  template<typename T>
  static std::vector<T> Bytes(HipBuffer<T>& b) {
    std::vector<T> out(b.bytes() / sizeof(T));
    CheckHip(hipMemcpy(out.data(), b.get(), b.bytes(), hipMemcpyDeviceToHost),
             "session download");
    return out;
  }
  bool Same(BatchSession& o) {
    const auto eq = [](auto a, auto b) {
      return a.size() == b.size() &&
             std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0;
    };
    return eq(Bytes(ring), Bytes(o.ring)) &&
           eq(Bytes(blocks), Bytes(o.blocks)) &&
           (format == KvCacheFormat::kF16
                ? eq(Bytes(f16k), Bytes(o.f16k)) &&
                      eq(Bytes(f16v), Bytes(o.f16v))
                : eq(Bytes(cache.k), Bytes(o.cache.k)) &&
                      eq(Bytes(cache.v), Bytes(o.cache.v)) &&
                      eq(Bytes(cache.ks), Bytes(o.cache.ks)) &&
                      eq(Bytes(cache.vs), Bytes(o.cache.vs)));
  }
  static constexpr std::uint32_t kRing = 4096;
  KvCacheFormat format;
  std::uint32_t position, n, context;
  TqCache cache;
  HipBuffer<__half> f16k, f16v;
  HipBuffer<float> ring;
  HipBuffer<__half> blocks;
  HipBuffer<std::uint32_t> control;  // position, pooled blocks
};

/// The multi-session stages must reproduce every session's single calls
/// bit for bit (projection preparation, rings, pooling, masks, attention),
/// including a cancelled session whose rows stay untouched.
void CheckAttentionBatch(KvCacheFormat format) {
  struct Shape {
    std::uint32_t position, n;
    bool live;
  };
  const Shape shapes[] = {{100, 1, true},    {2044, 3, true},
                          {4096, 8, true},   {32765, 2, true},
                          {50000, 4, false}, {131069, 5, true}};
  constexpr std::uint32_t stride = 2 * (kQWidth + kKvWidth);
  constexpr std::uint32_t kMaxBlocks = (131074 / kRatio + 31) / 32 * 32;
  constexpr std::uint32_t kMaskWords = kMaxBlocks / 32;
  std::uint32_t rows = 0;
  for (const Shape& s : shapes)
    rows += s.n;
  HipBuffer<float> packed(std::size_t(rows) * stride),
      iq(std::size_t(rows) * 512), ik(std::size_t(rows) * 128), gamma(3 * kDim);
  Upload(&packed, MakeValues(std::size_t(rows) * stride, 3, 4.0F));
  Upload(&iq, MakeValues(std::size_t(rows) * 512, 5, 1.0F));
  Upload(&ik, MakeValues(std::size_t(rows) * 128, 6, 1.0F));
  auto g = MakeValues(3 * kDim, 8, 0.5F);
  for (auto& x : g)
    x += 1.0F;
  Upload(&gamma, g);
  // Per-route outputs: queries, gates, masks, scores, context, partials.
  struct Route {
    explicit Route(std::uint32_t rows)
        : q(std::size_t(rows) * kQWidth),
          gate(std::size_t(rows) * kQWidth),
          iq(std::size_t(rows) * 512),
          mask(std::size_t(rows) * kMaskWords),
          scores(std::size_t(512) * kMaxBlocks),
          ctx(std::size_t(rows) * kQWidth),
          partials(std::size_t(64) * 8 * kHeads * (kDim + 2)) {}
    HipBuffer<float> q, gate, iq;
    HipBuffer<std::uint32_t> mask;
    HipBuffer<float> scores, ctx, partials;
  } single(rows), batch(rows);
  std::vector<std::unique_ptr<BatchSession>> a, b;
  std::uint32_t seed = 40;
  for (const Shape& s : shapes) {
    a.push_back(std::make_unique<BatchSession>(format, s.position, s.n, seed));
    b.push_back(std::make_unique<BatchSession>(format, s.position, s.n, seed));
    seed += 10;
  }
  CheckHip(
      hipMemcpy(single.iq.get(), iq.get(), iq.bytes(), hipMemcpyDeviceToDevice),
      "iq copy");
  CheckHip(
      hipMemcpy(batch.iq.get(), iq.get(), iq.bytes(), hipMemcpyDeviceToDevice),
      "iq copy");
  const bool tq = format != KvCacheFormat::kF16;
  // Single-session reference, following Executor::Attention.
  std::uint32_t row = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    BatchSession& s = *a[i];
    const std::uint32_t n = s.n;
    if (shapes[i].live) {
      const std::uint32_t* pos = s.control.get();
      if (!q::PrepareAttention(
              packed.get() + std::size_t(row) * stride, stride, gamma.get(),
              gamma.get() + kDim, single.q.get() + std::size_t(row) * kQWidth,
              single.gate.get() + std::size_t(row) * kQWidth, s.View(), n,
              kHeads, kKvHeads, kDim, 64, pos, 1e7F, 1e-6F, nullptr))
        throw std::runtime_error("batch reference preparation rejected");
      q::StoreRows(ik.get() + std::size_t(row) * 128, s.ring.get(), n, 128, pos,
                   BatchSession::kRing, nullptr);
      const std::uint32_t end = s.position + n;
      const bool sparse = end > 2048;
      float* qi = single.iq.get() + std::size_t(row) * 512;
      q::RmsNormRows(qi, gamma.get() + 2 * kDim, qi, n * 4, 128, 1, 1e-6F,
                     nullptr);
      if (sparse) {
        q::Rope(qi, n, 4, 128, 64, pos, 1e7F, nullptr);
        const std::uint32_t pool = end / kRatio - s.position / kRatio;
        q::PoolIndexerBlocks(s.ring.get(), gamma.get() + 2 * kDim,
                             s.blocks.get(), s.control.get() + 1, pos, n, pool,
                             kRatio, 128, 64, 1e7F, 1e-6F, BatchSession::kRing,
                             nullptr);
        q::SelectBlocks(qi, s.blocks.get(),
                        single.mask.get() + std::size_t(row) * kMaskWords,
                        single.scores.get(), n, pos, 0, 4, 128, kRatio, 512,
                        kMaskWords, kMaxBlocks, nullptr);
      }
      float* qr = single.q.get() + std::size_t(row) * kQWidth;
      if (tq)
        q::RotateHeads(qr, n * kHeads, nullptr);
      float* out = single.ctx.get() + std::size_t(row) * kQWidth;
      q::Attention(
          qr, s.View(),
          sparse ? single.mask.get() + std::size_t(row) * kMaskWords : nullptr,
          kMaskWords, out, single.partials.get(), q::AttentionSplits(n), n, pos,
          kHeads, kKvHeads, kDim, kRatio, nullptr);
    }
    row += n;
  }
  // The batch: one descriptor table, one launch per stage.
  q::AttentionBatchItem* items = nullptr;
  CheckHip(hipHostMalloc(&items, sizeof(q::AttentionBatchItem) * b.size(),
                         hipHostMallocMapped),
           "batch table");
  q::AttentionBatchItem* device_items = nullptr;
  CheckHip(hipHostGetDevicePointer(reinterpret_cast<void**>(&device_items),
                                   items, 0),
           "batch table device");
  row = 0;
  std::uint32_t partial = 0, max_rows = 0, max_pool = 0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    BatchSession& s = *b[i];
    const std::uint32_t end = s.position + s.n;
    const bool sparse = end > 2048;
    items[i] = q::AttentionBatchItem{
        .kv = s.View(),
        .index_k = s.ring.get(),
        .block_k = s.blocks.get(),
        .position = s.control.get(),
        .first_block = s.control.get() + 1,
        .rope = nullptr,
        .index_capacity = BatchSession::kRing,
        .row = row,
        .rows = shapes[i].live ? s.n : 0,
        .pool = sparse ? end / kRatio - s.position / kRatio : 0,
        .sparse = sparse ? 1U : 0U,
        .splits = q::AttentionSplits(s.n),
        .partial = partial};
    if (shapes[i].live) {
      partial += s.n * kHeads * items[i].splits;
      max_rows = std::max(max_rows, s.n);
      max_pool = std::max(max_pool, items[i].pool);
    }
    row += s.n;
  }
  const auto count = static_cast<std::uint32_t>(b.size());
  q::PrepareAttentionBatch(packed.get(), stride, gamma.get(),
                           gamma.get() + kDim, batch.q.get(), batch.gate.get(),
                           format, device_items, count, max_rows, kHeads,
                           kKvHeads, kDim, 64, 1e7F, 1e-6F, nullptr);
  q::StoreIndexerBatch(ik.get(), gamma.get() + 2 * kDim, device_items, count,
                       max_rows, max_pool, kRatio, 128, 64, 1e7F, 1e-6F,
                       nullptr);
  q::RmsNormRows(batch.iq.get(), gamma.get() + 2 * kDim, batch.iq.get(),
                 rows * 4, 128, 1, 1e-6F, nullptr);
  if (!q::SelectBlocksBatch(batch.iq.get(), batch.mask.get(),
                            batch.scores.get(), device_items, count, max_rows,
                            rows, 512, 4, 128, 64, 1e7F, kRatio, 512,
                            kMaskWords, kMaxBlocks, nullptr))
    throw std::runtime_error("multi-session block selection rejected");
  if (tq)
    q::RotateHeads(batch.q.get(), rows * kHeads, nullptr);
  q::AttentionBatch(batch.q.get(), format, device_items, count, max_rows,
                    batch.mask.get(), kMaskWords, batch.ctx.get(),
                    batch.partials.get(), kHeads, kKvHeads, kDim, kRatio,
                    nullptr);
  CheckHip(hipDeviceSynchronize(), "multi-session attention");
  (void)hipHostFree(items);
  const auto same_rows = [&](HipBuffer<float>& x, HipBuffer<float>& y,
                             std::size_t width, const char* what) {
    const auto u = BatchSession::Bytes(x), v = BatchSession::Bytes(y);
    std::uint32_t r = 0;
    for (std::size_t i = 0; i < b.size(); ++i) {
      const std::size_t first = std::size_t(r) * width;
      const std::size_t size = std::size_t(shapes[i].n) * width;
      if (shapes[i].live &&
          std::memcmp(u.data() + first, v.data() + first, size * 4) != 0)
        throw std::runtime_error(std::string("multi-session ") + what +
                                 " differs for session " + std::to_string(i));
      r += shapes[i].n;
    }
  };
  same_rows(single.q, batch.q, kQWidth, "queries");
  same_rows(single.gate, batch.gate, kQWidth, "gates");
  same_rows(single.ctx, batch.ctx, kQWidth, "attention");
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!a[i]->Same(*b[i]))
      throw std::runtime_error("multi-session caches differ for session " +
                               std::to_string(i));
  }
  std::cout << "multi-session attention " << (tq ? FormatName(format) : "f16")
            << ": bit-exact against single-session calls\n";
}

/// Packed prompt rows of several sessions (a fused prefill step holds up to
/// 2048 of them) must select their blocks inside the score scratch. The
/// executor sizes that scratch for 512 rows at the model's full 262144
/// context, so at a 131072-token session context it holds 1024 rows of
/// max_blocks floats: a deep session whose packed rows run past that wrote
/// its scores beyond the allocation (GPU memory corruption or an illegal
/// memory access that surfaced at the next host sync). The masks must stay
/// bit-exact against the single-session calls and the scratch's guard
/// region untouched.
void CheckSelectBatchRowCapacity() {
  struct Shape {
    std::uint32_t position, n;
  };
  const Shape shapes[] = {{5000, 1100}, {3000, 900}};
  constexpr std::uint32_t kMaxBlocks = (131074 / kRatio + 31) / 32 * 32;
  constexpr std::uint32_t kMaskWords = kMaxBlocks / 32;
  constexpr std::uint32_t kScoreRows = 1024;  // 512 x (262144 / 131072)
  constexpr std::uint32_t kGuardRows = 1024;
  constexpr std::uint32_t kSentinel = 0xFFFFFFFFU;
  std::uint32_t rows = 0;
  for (const Shape& s : shapes)
    rows += s.n;
  HipBuffer<float> iq(std::size_t(rows) * 512);
  Upload(&iq, MakeValues(std::size_t(rows) * 512, 11, 1.0F));
  HipBuffer<float> single_iq(std::size_t(rows) * 512);
  CheckHip(
      hipMemcpy(single_iq.get(), iq.get(), iq.bytes(), hipMemcpyDeviceToDevice),
      "iq copy");
  HipBuffer<std::uint32_t> single_mask(std::size_t(rows) * kMaskWords),
      batch_mask(std::size_t(rows) * kMaskWords);
  HipBuffer<float> single_scores(std::size_t(256) * kMaxBlocks);
  HipBuffer<float> scores(std::size_t(kScoreRows + kGuardRows) * kMaxBlocks);
  CheckHip(hipMemset(scores.get(), 0xFF, scores.bytes()), "score sentinel");

  std::vector<std::unique_ptr<BatchSession>> a, b;
  for (std::uint32_t i = 0; i < 2; ++i) {
    a.push_back(std::make_unique<BatchSession>(
        KvCacheFormat::kF16, shapes[i].position, shapes[i].n, 70 + i));
    b.push_back(std::make_unique<BatchSession>(
        KvCacheFormat::kF16, shapes[i].position, shapes[i].n, 70 + i));
  }
  // Single-session reference, following Executor::Attention (256-row chunks).
  std::uint32_t row = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    BatchSession& s = *a[i];
    float* qi = single_iq.get() + std::size_t(row) * 512;
    q::Rope(qi, s.n, 4, 128, 64, s.control.get(), 1e7F, nullptr);
    for (std::uint32_t t0 = 0; t0 < s.n; t0 += 256)
      q::SelectBlocks(qi + std::size_t(t0) * 512, s.blocks.get(),
                      single_mask.get() + std::size_t(row + t0) * kMaskWords,
                      single_scores.get(), std::min(256U, s.n - t0),
                      s.control.get(), t0, 4, 128, kRatio, 512, kMaskWords,
                      kMaxBlocks, nullptr);
    row += s.n;
  }
  q::AttentionBatchItem* items = nullptr;
  CheckHip(hipHostMalloc(&items, sizeof(q::AttentionBatchItem) * b.size(),
                         hipHostMallocMapped),
           "batch table");
  q::AttentionBatchItem* device_items = nullptr;
  CheckHip(hipHostGetDevicePointer(reinterpret_cast<void**>(&device_items),
                                   items, 0),
           "batch table device");
  row = 0;
  std::uint32_t max_rows = 0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    BatchSession& s = *b[i];
    items[i] = q::AttentionBatchItem{.block_k = s.blocks.get(),
                                     .position = s.control.get(),
                                     .row = row,
                                     .rows = s.n,
                                     .sparse = 1U};
    max_rows = std::max(max_rows, s.n);
    row += s.n;
  }
  if (!q::SelectBlocksBatch(iq.get(), batch_mask.get(), scores.get(),
                            device_items, static_cast<std::uint32_t>(b.size()),
                            max_rows, rows, kScoreRows, 4, 128, 64, 1e7F,
                            kRatio, 512, kMaskWords, kMaxBlocks, nullptr))
    throw std::runtime_error("packed block selection rejected");
  CheckHip(hipDeviceSynchronize(), "packed block selection");
  (void)hipHostFree(items);

  std::vector<std::uint32_t> guard(std::size_t(kGuardRows) * kMaxBlocks);
  CheckHip(
      hipMemcpy(guard.data(),
                scores.get() + std::size_t(kScoreRows) * kMaxBlocks,
                guard.size() * sizeof(std::uint32_t), hipMemcpyDeviceToHost),
      "guard download");
  std::size_t clobbered = 0;
  for (const std::uint32_t word : guard)
    clobbered += word != kSentinel;
  if (clobbered != 0)
    throw std::runtime_error(
        "packed block selection wrote " + std::to_string(clobbered) +
        " score words past its " + std::to_string(kScoreRows) + "-row scratch");
  std::vector<std::uint32_t> want(single_mask.bytes() / 4),
      got(batch_mask.bytes() / 4);
  CheckHip(hipMemcpy(want.data(), single_mask.get(), single_mask.bytes(),
                     hipMemcpyDeviceToHost),
           "mask download");
  CheckHip(hipMemcpy(got.data(), batch_mask.get(), batch_mask.bytes(),
                     hipMemcpyDeviceToHost),
           "mask download");
  if (want != got)
    throw std::runtime_error(
        "packed block selection masks differ from the single-session calls");
  std::cout << "packed block selection: " << rows << " rows through a "
            << kScoreRows
            << "-row score scratch, bit-exact, guard region untouched\n";
}

/// qwen35moe geometry (16 query heads over 2 KV heads): the grouped kernel
/// against the per-head kernel (one query head per KV head per call, which
/// stays on AttentionKernel), and its multi-session calls bit-exact against
/// single-session calls.
void CheckGqaAttention(KvCacheFormat format) {
  constexpr std::uint32_t kGqaHeads = 16, kGroup = kGqaHeads / kKvHeads;
  constexpr std::uint32_t kWidth = kGqaHeads * kDim;
  struct Shape {
    std::uint32_t position, n;
  };
  const Shape shapes[] = {{0, 1},    {15, 1},   {16, 2},   {100, 1},
                          {2047, 3}, {5000, 1}, {32765, 2}};
  std::uint32_t rows = 0;
  for (const Shape& s : shapes)
    rows += s.n;
  HipBuffer<float> qs(std::size_t(rows) * kWidth);
  Upload(&qs, MakeValues(std::size_t(rows) * kWidth, 77, 2.0F));
  HipBuffer<float> single(std::size_t(rows) * kWidth),
      batch(std::size_t(rows) * kWidth), ref(std::size_t(rows) * kWidth);
  HipBuffer<float> partials(std::size_t(rows) * kGqaHeads * 8 * (kDim + 2));
  std::vector<std::unique_ptr<BatchSession>> sessions;
  std::uint32_t seed = 90;
  for (const Shape& s : shapes)
    sessions.push_back(
        std::make_unique<BatchSession>(format, s.position, s.n, seed += 10));
  // Reference: query heads g and kGroup + g as one two-head call per g.
  HipBuffer<float> q2(std::size_t(8) * 2 * kDim), o2(std::size_t(8) * 2 * kDim);
  HipBuffer<float> p2(std::size_t(8) * 2 * 8 * (kDim + 2));
  std::uint32_t row = 0;
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    BatchSession& s = *sessions[i];
    for (std::uint32_t g = 0; g < kGroup; ++g) {
      for (std::uint32_t r = 0; r < s.n; ++r)
        for (std::uint32_t h = 0; h < 2; ++h)
          CheckHip(hipMemcpy(q2.get() + (std::size_t(r) * 2 + h) * kDim,
                             qs.get() + (std::size_t(row + r) * kGqaHeads +
                                         h * kGroup + g) *
                                            kDim,
                             kDim * 4, hipMemcpyDeviceToDevice),
                   "reference queries");
      q::Attention(q2.get(), s.View(), nullptr, 0, o2.get(), p2.get(),
                   q::AttentionSplits(s.n), s.n, s.control.get(), 2, kKvHeads,
                   kDim, kRatio, nullptr);
      for (std::uint32_t r = 0; r < s.n; ++r)
        for (std::uint32_t h = 0; h < 2; ++h)
          CheckHip(hipMemcpy(ref.get() + (std::size_t(row + r) * kGqaHeads +
                                          h * kGroup + g) *
                                             kDim,
                             o2.get() + (std::size_t(r) * 2 + h) * kDim,
                             kDim * 4, hipMemcpyDeviceToDevice),
                   "reference output");
    }
    q::Attention(qs.get() + std::size_t(row) * kWidth, s.View(), nullptr, 0,
                 single.get() + std::size_t(row) * kWidth, partials.get(),
                 q::AttentionSplits(s.n), s.n, s.control.get(), kGqaHeads,
                 kKvHeads, kDim, kRatio, nullptr);
    row += s.n;
  }
  q::AttentionBatchItem* items = nullptr;
  CheckHip(hipHostMalloc(&items, sizeof(q::AttentionBatchItem) * 8,
                         hipHostMallocMapped),
           "gqa table");
  q::AttentionBatchItem* device_items = nullptr;
  CheckHip(hipHostGetDevicePointer(reinterpret_cast<void**>(&device_items),
                                   items, 0),
           "gqa table device");
  row = 0;
  std::uint32_t partial = 0, max_rows = 0;
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    BatchSession& s = *sessions[i];
    items[i] = q::AttentionBatchItem{.kv = s.View(),
                                     .position = s.control.get(),
                                     .row = row,
                                     .rows = s.n,
                                     .splits = q::AttentionSplits(s.n),
                                     .partial = partial};
    partial += s.n * kGqaHeads * items[i].splits;
    max_rows = std::max(max_rows, s.n);
    row += s.n;
  }
  q::AttentionBatch(qs.get(), format, device_items,
                    static_cast<std::uint32_t>(sessions.size()), max_rows,
                    nullptr, 0, batch.get(), partials.get(), kGqaHeads,
                    kKvHeads, kDim, kRatio, nullptr);
  CheckHip(hipDeviceSynchronize(), "grouped attention");
  (void)hipHostFree(items);
  const std::size_t count = std::size_t(rows) * kWidth;
  const auto a = Download(&single, count), b = Download(&batch, count),
             r = Download(&ref, count);
  if (std::memcmp(a.data(), b.data(), count * 4) != 0)
    throw std::runtime_error("grouped attention: batch differs from single");
  double err = 0, norm = 0, worst = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const double d = double(a[i]) - r[i];
    err += d * d;
    norm += double(r[i]) * r[i];
    worst = std::max(worst, std::abs(d));
  }
  const double rms = std::sqrt(norm / count), rel = std::sqrt(err / norm);
  std::cout << "grouped attention "
            << (format == KvCacheFormat::kF16 ? "f16" : FormatName(format))
            << ": rel RMS " << rel << ", max |err| " << worst / rms << " RMS\n";
  if (!(rel < 4e-3 && worst < 4e-2 * rms))
    throw std::runtime_error("grouped attention differs from per-head");
}

/// Average microseconds per call of `run` (after a warm-up call).
template<typename Run>
double TimeUs(Run&& run, int reps) {
  run();
  CheckHip(hipDeviceSynchronize(), "bench warm-up");
  hipEvent_t a, b;
  CheckHip(hipEventCreate(&a), "event");
  CheckHip(hipEventCreate(&b), "event");
  CheckHip(hipEventRecord(a, nullptr), "event");
  for (int i = 0; i < reps; ++i)
    run();
  CheckHip(hipEventRecord(b, nullptr), "event");
  CheckHip(hipEventSynchronize(b), "event");
  float ms = 0.0F;
  CheckHip(hipEventElapsedTime(&ms, a, b), "event");
  (void)hipEventDestroy(a);
  (void)hipEventDestroy(b);
  return 1000.0 * ms / reps;
}

/// Decode (split-K, <= 8 rows) and wide sparse prefill timings per format
/// over 512-block (2048-token) selections. Caches exceed the 32 MiB MALL.
void Bench() {
  for (const std::uint32_t depth : {32768U, 131072U}) {
    const std::uint32_t n_kv = depth + 2048;
    const std::uint32_t max_blocks = (n_kv + kRatio - 1) / kRatio;
    const std::uint32_t mask_words = (max_blocks + 31) / 32;
    HipBuffer<__half> f16k(std::size_t(n_kv) * kKvWidth),
        f16v(std::size_t(n_kv) * kKvWidth);
    TqCache tq4(KvCacheFormat::kTq4, n_kv), tq3(KvCacheFormat::kTq3, n_kv);
    {
      // Content does not change the timing; quantize random rows once.
      HipBuffer<float> rows(std::size_t(n_kv) * kKvWidth);
      Upload(&rows, MakeValues(std::size_t(n_kv) * kKvWidth, 3, 1.0F));
      HipBuffer<std::uint32_t> zero(1);
      q::StoreKv(rows.get(), f16k.get(), n_kv, kKvWidth, zero.get(), nullptr);
      q::StoreKv(rows.get(), f16v.get(), n_kv, kKvWidth, zero.get(), nullptr);
      q::QuantizeKv(rows.get(), rows.get(), tq4.View(), n_kv, kKvHeads,
                    zero.get(), nullptr);
      q::QuantizeKv(rows.get(), rows.get(), tq3.View(), n_kv, kKvHeads,
                    zero.get(), nullptr);
      CheckHip(hipDeviceSynchronize(), "bench caches");
    }
    constexpr std::uint32_t kWide = 2048;
    std::vector<std::uint32_t> mask(std::size_t(kWide) * mask_words, 0U);
    std::uint32_t state = 99;
    for (std::uint32_t t = 0; t < kWide; ++t) {
      const std::uint32_t complete = (depth + t + 1) / kRatio;
      for (std::uint32_t s = 0; s < 512;) {
        const std::uint32_t block = NextRandom(&state) % complete;
        auto& word = mask[std::size_t(t) * mask_words + block / 32];
        if ((word & (1U << (block % 32))) == 0) {
          word |= 1U << (block % 32);
          ++s;
        }
      }
    }
    HipBuffer<std::uint32_t> dmask(mask.size()), dpos(1);
    Upload(&dmask, mask);
    Upload(&dpos, std::vector<std::uint32_t>{depth});
    HipBuffer<float> dq(std::size_t(kWide) * kQWidth),
        dgate(std::size_t(kWide) * kQWidth), dout(std::size_t(kWide) * kQWidth),
        partials(std::size_t(8) * kHeads * 8 * (kDim + 2));
    Upload(&dq, MakeValues(std::size_t(kWide) * kQWidth, 5, 1.0F));
    const q::KvView views[] = {
        q::KvView{KvCacheFormat::kF16, f16k.get(), f16v.get()}, tq4.View(),
        tq3.View()};
    const char* names[] = {"f16", "tq4", "tq3"};
    for (const std::uint32_t rows : {1U, 4U, 8U}) {
      std::cout << "bench decode depth=" << depth << " rows=" << rows;
      for (int f = 0; f < 3; ++f) {
        const bool tq = views[f].format != KvCacheFormat::kF16;
        const double us = TimeUs(
            [&] {
              if (tq)
                q::RotateHeads(dq.get(), rows * kHeads, nullptr);
              q::Attention(dq.get(), views[f], dmask.get(), mask_words,
                           dout.get(), partials.get(), 8, rows, dpos.get(),
                           kHeads, kKvHeads, kDim, kRatio, nullptr);
              if (tq)
                q::UnrotateHeads(dout.get(), dgate.get(), rows * kHeads,
                                 nullptr);
              else
                q::SigmoidMul(dout.get(), dgate.get(),
                              std::size_t(rows) * kQWidth, nullptr);
            },
            200);
        std::cout << ' ' << names[f] << ' ' << us << "us";
      }
      std::cout << '\n';
    }
    std::cout << "bench prefill depth=" << depth << " rows=" << kWide;
    for (int f = 0; f < 3; ++f) {
      const bool tq = views[f].format != KvCacheFormat::kF16;
      const double us = TimeUs(
          [&] {
            if (tq)
              q::RotateHeads(dq.get(), kWide * kHeads, nullptr);
            (void)q::WmmaCausalAttention(dq.get(), tq ? nullptr : dgate.get(),
                                         views[f], dmask.get(), mask_words,
                                         dout.get(), kWide, depth, kHeads,
                                         kKvHeads, kDim, kRatio, nullptr);
            if (tq)
              q::UnrotateHeads(dout.get(), dgate.get(), kWide * kHeads,
                               nullptr);
          },
          5);
      std::cout << ' ' << names[f] << ' ' << us / 1000.0 << "ms";
    }
    std::cout << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc > 1 && std::string(argv[1]) == "--bench") {
      Bench();
      return 0;
    }
    for (const auto format :
         {KvCacheFormat::kF16, KvCacheFormat::kTq4, KvCacheFormat::kTq3})
      CheckAttentionBatch(format);
    CheckSelectBatchRowCapacity();
    for (const auto format :
         {KvCacheFormat::kF16, KvCacheFormat::kTq4, KvCacheFormat::kTq3})
      CheckGqaAttention(format);
    for (const auto format : {KvCacheFormat::kTq4, KvCacheFormat::kTq3}) {
      CheckTqStore(format);
      CheckTqPreparation(format, 1, 0, false);
      CheckTqPreparation(format, 9, 4093, false);
      CheckTqPreparation(format, 65, 131069, true);
      std::uint32_t seed = 0x51A7U;
      struct TqCase {
        std::uint32_t n;
        std::uint32_t start;
        bool masked;
        bool wmma;
      };
      const TqCase tq_cases[] = {
          {1, 100, false, false},   {7, 2040, false, false},
          {8, 4096, true, false},   {9, 32765, true, false},
          {1, 131069, true, false}, {100, 0, false, true},
          {64, 37, false, true},    {96, 4096, true, true},
          {7, 131069, true, true}};
      for (const TqCase& c : tq_cases)
        CheckTqAttention(format, c.n, c.start, c.masked, c.wmma, seed++);
    }
    std::cout << "TurboQuant KV checks passed\n";
    CheckChunks(136, 94);
    CheckChunks(2048, 1025);
    CheckPreparation(1, 0, 64);
    CheckPreparation(8, 4096, 64);
    CheckPreparation(65, 131069, 64);
    CheckPreparation(7, 131069, 256);
    struct Case {
      std::uint32_t n_tokens;
      std::uint32_t start_pos;
      bool masked;
      bool sparse;
      bool bounded{false};
    };
    const Case cases[] = {
        {4, 4096, false, false},      {3, 9000, true, true},
        {100, 0, false, false},       {64, 37, false, false},
        {100, 0, true, false},        {77, 51, true, false},
        {96, 4096, true, true},       {70, 8000, true, true},
        {5, 32765, true, true, true}, {7, 65533, true, true, true},
        {5, 131069, true, true},      {7, 131069, true, true, true},
        {17, 2047, true, true, true}};
    bool ok = true;
    std::uint32_t seed = 0x1234ABCDU;
    for (const Case& c : cases) {
      const double worst = Compare(c.n_tokens, c.start_pos, c.masked, seed++,
                                   c.sparse, c.bounded);
      std::cout << "WMMA attention n=" << c.n_tokens << " start=" << c.start_pos
                << (c.masked ? (c.sparse ? " sparse" : " masked") : " dense")
                << " worst absolute error " << worst << '\n';
      // The reference accumulates FP32 probabilities; the WMMA route rounds
      // Q and P to FP16, so the contract is a small absolute envelope on
      // values of order one.
      ok = ok && worst < 2e-2;
    }
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

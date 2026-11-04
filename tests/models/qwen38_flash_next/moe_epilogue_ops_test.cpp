#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/config.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

// The model's MoE combine geometry: top-10 slots over a 2,560-wide hidden
// state, router row stride num_experts + 1.
constexpr std::uint32_t kTokens = 37;
constexpr std::uint32_t kSlots = 10;
constexpr std::uint32_t kHidden = 2560;
constexpr std::uint32_t kGateStride = 513;

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
                              float scale, float offset = 0.0F) {
  std::vector<float> values(count);
  for (float& value : values) {
    value = offset +
            scale *
                static_cast<float>(
                    static_cast<int>(NextRandom(&seed) & 0xFFFFU) - 32768) /
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

template<typename T>
std::vector<T> Download(HipBuffer<T>* source, std::size_t count) {
  std::vector<T> values(count);
  CheckHip(hipMemcpy(values.data(), source->get(), count * sizeof(T),
                     hipMemcpyDeviceToHost),
           "download");
  return values;
}

// Steering adapters: two unit directions; rows alternate none / 1 / 2.
constexpr std::uint32_t kAdapters = 2;
constexpr float kAlphas[kAdapters] = {1.9553F, 0.5F};

std::vector<std::uint8_t> AdapterRows() {
  std::vector<std::uint8_t> rows(kTokens);
  for (std::uint32_t t = 0; t < kTokens; ++t)
    rows[t] = static_cast<std::uint8_t>(t % 3);
  return rows;
}

std::vector<float> UnitDirections() {
  auto r = MakeValues(std::size_t{kAdapters} * kHidden, 0x5EEDF00DU, 1.0F);
  for (std::uint32_t a = 0; a < kAdapters; ++a) {
    double norm = 0.0;
    for (std::uint32_t i = 0; i < kHidden; ++i)
      norm += static_cast<double>(r[a * kHidden + i]) * r[a * kHidden + i];
    for (std::uint32_t i = 0; i < kHidden; ++i)
      r[a * kHidden + i] =
          static_cast<float>(r[a * kHidden + i] / std::sqrt(norm));
  }
  return r;
}

// MoE outputs of layer 0: adapter 1 ablates only the gated shared expert
// there (its routed layers exclude 0), adapter 2 the whole output.
const std::vector<std::uint64_t> kRoutedLayers = {~std::uint64_t{1},
                                                  ~std::uint64_t{0}};

float Sigmoid(float x) {
  return 1.0F / (1.0F + std::exp(-x));
}

// Steer against a double-precision reference, on attention outputs and on
// layer-0 MoE outputs; rows without an adapter must keep their bits.
bool CheckSteer(bool moe) {
  const auto y = MakeValues(std::size_t{kTokens} * kHidden, 0x0DDBA11U, 2.0F);
  const auto shared =
      MakeValues(std::size_t{kTokens} * kHidden, 0xDEADBEEFU, 1.0F);
  const auto gate =
      MakeValues(std::size_t{kTokens} * kGateStride, 0xC0FFEE11U, 3.0F);
  const auto r = UnitDirections();
  const auto rows = AdapterRows();
  HipBuffer<float> d_y(y.size()), d_r(r.size()), d_alpha(kAdapters),
      d_shared(shared.size()), d_gate(gate.size());
  HipBuffer<std::uint8_t> d_rows(rows.size());
  HipBuffer<std::uint64_t> d_layers(kAdapters);
  Upload(&d_y, y);
  Upload(&d_r, r);
  Upload(&d_alpha, std::vector<float>(kAlphas, kAlphas + kAdapters));
  Upload(&d_rows, rows);
  Upload(&d_shared, shared);
  Upload(&d_gate, gate);
  Upload(&d_layers, kRoutedLayers);
  q::SteerRows steer{d_rows.get(), d_r.get(), d_alpha.get()};
  if (moe)
    steer = {d_rows.get(),   d_r.get(),    d_alpha.get(), d_layers.get(), 0,
             d_shared.get(), d_gate.get(), kGateStride};
  q::Steer(d_y.get(), steer, kTokens, kHidden, nullptr);
  CheckHip(hipDeviceSynchronize(), "steer synchronization");
  const auto out = Download(&d_y, y.size());
  double worst = 0.0, residual = 0.0;
  bool untouched = true;
  for (std::uint32_t t = 0; t < kTokens; ++t) {
    const float* in = y.data() + std::size_t{t} * kHidden;
    const float* got = out.data() + std::size_t{t} * kHidden;
    if (rows[t] == 0) {
      untouched &= std::equal(in, in + kHidden, got);
      continue;
    }
    const float* d = r.data() + static_cast<std::size_t>(rows[t] - 1) * kHidden;
    const bool shared_only = moe && rows[t] == 1;
    const float* p =
        shared_only ? shared.data() + std::size_t{t} * kHidden : in;
    const double scale =
        shared_only ? Sigmoid(gate[std::size_t{t} * kGateStride]) : 1.0;
    double dot = 0.0, after = 0.0, before = 0.0;
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      dot += scale * d[i] * p[i];
      before += static_cast<double>(d[i]) * in[i];
    }
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      const double want = in[i] - kAlphas[rows[t] - 1] * dot * d[i];
      worst = std::max(worst, std::abs(want - got[i]));
      after += static_cast<double>(d[i]) * got[i];
    }
    // alpha = 1 would remove the component; alpha a leaves (1 - a) of it.
    residual = std::max(
        residual, std::abs(after - (before - kAlphas[rows[t] - 1] * dot)));
  }
  std::cout << (moe ? "MoE " : "") << "steer worst absolute error " << worst
            << " projection error " << residual
            << " untouched_rows_exact=" << untouched << '\n';
  return untouched && worst < 1e-4 && residual < 1e-3;
}

// The steered fused MoE combine keeps the plain kernel's bits for rows
// without an adapter and matches MoeEpilogueVec4F16 + Steer + HcCombineF16
// on adapter rows, for whole and shared-expert-only ablation.
bool CheckSteeredMoeCombine() {
  constexpr std::uint32_t kStreams = 4;
  constexpr std::size_t kExpert = std::size_t{kTokens} * kSlots * kHidden;
  constexpr std::size_t kRes = std::size_t{kTokens} * kStreams * kHidden;
  std::vector<__half> expert(kExpert);
  {
    const auto values = MakeValues(kExpert, 0x1234ABCDU, 1.0F);
    for (std::size_t i = 0; i < kExpert; ++i)
      expert[i] = __float2half(values[i]);
  }
  const auto weights =
      MakeValues(std::size_t{kTokens} * kSlots, 0xBADC0FFEU, 0.5F, 0.5F);
  const auto shared =
      MakeValues(std::size_t{kTokens} * kHidden, 0xDEADBEEFU, 1.0F);
  const auto gate =
      MakeValues(std::size_t{kTokens} * kGateStride, 0xC0FFEE11U, 3.0F);
  const auto inject =
      MakeValues(std::size_t{kTokens} * kStreams, 0xFEEDFACEU, 2.0F);
  const auto gamma =
      MakeValues(std::size_t{kStreams} * kHidden, 0xA11CEU, 0.5F, 1.0F);
  const auto res = MakeValues(kRes, 0xB0BAFE77U, 4.0F);
  const auto r = UnitDirections();
  const auto rows = AdapterRows();
  HipBuffer<__half> d_expert(kExpert);
  HipBuffer<float> d_weights(weights.size()), d_shared(shared.size()),
      d_gate(gate.size()), d_inject(inject.size()), d_gamma(gamma.size()),
      d_r(r.size()), d_alpha(kAdapters), d_block(shared.size());
  HipBuffer<std::uint8_t> d_rows(rows.size());
  HipBuffer<std::uint64_t> d_layers(kAdapters);
  Upload(&d_layers, kRoutedLayers);
  Upload(&d_expert, expert);
  Upload(&d_weights, weights);
  Upload(&d_shared, shared);
  Upload(&d_gate, gate);
  Upload(&d_inject, inject);
  Upload(&d_gamma, gamma);
  Upload(&d_r, r);
  Upload(&d_alpha, std::vector<float>(kAlphas, kAlphas + kAdapters));
  Upload(&d_rows, rows);
  const q::SteerRows steer{d_rows.get(),   d_r.get(),  d_alpha.get(),
                           d_layers.get(), 0,          d_shared.get(),
                           d_gate.get(),   kGateStride};
  const std::size_t q8_bytes = q::Q8TiledBytes(kTokens, kStreams * kHidden);
  // 0: plain fused, 1: steered fused, 2: unfused epilogue + Steer.
  std::vector<float> out_res[3];
  std::vector<__half> out_xn[3];
  for (int variant = 0; variant < 3; ++variant) {
    HipBuffer<float> d_res(kRes);
    HipBuffer<__half> d_xn(kRes);
    HipBuffer<std::uint8_t> d_q8(q8_bytes);
    Upload(&d_res, res);
    if (variant < 2) {
      if (!q::HcCombineMoeF16(d_res.get(), d_expert.get(), d_weights.get(),
                              d_shared.get(), d_gate.get(), kGateStride, kSlots,
                              d_inject.get(), 1, d_gamma.get(), d_xn.get(),
                              d_q8.get(), kTokens, kHidden, kStreams, 1e-6F,
                              nullptr, variant == 1 ? &steer : nullptr))
        throw std::runtime_error("fused MoE combine rejected the geometry");
    } else {
      q::MoeEpilogueVec4F16(d_expert.get(), d_weights.get(), d_shared.get(),
                            d_gate.get(), kGateStride, d_block.get(), kTokens,
                            kSlots, kHidden, nullptr);
      q::Steer(d_block.get(), steer, kTokens, kHidden, nullptr);
      q::HcCombineF16(d_res.get(), d_block.get(), d_inject.get(), 1,
                      d_gamma.get(), d_xn.get(), d_q8.get(), kTokens, kHidden,
                      kStreams, 1e-6F, nullptr);
    }
    CheckHip(hipDeviceSynchronize(), "MoE combine synchronization");
    out_res[variant] = Download(&d_res, kRes);
    out_xn[variant] = Download(&d_xn, kRes);
  }
  bool plain_rows_exact = true;
  double worst = 0.0, scale = 0.0;
  const std::size_t row = std::size_t{kStreams} * kHidden;
  for (std::uint32_t t = 0; t < kTokens; ++t) {
    const std::size_t at = t * row;
    if (rows[t] == 0) {
      plain_rows_exact &=
          std::equal(out_res[0].begin() + at, out_res[0].begin() + at + row,
                     out_res[1].begin() + at) &&
          std::memcmp(out_xn[0].data() + at, out_xn[1].data() + at,
                      row * sizeof(__half)) == 0;
      continue;
    }
    for (std::size_t i = at; i < at + row; ++i) {
      worst = std::max(
          worst, std::abs(static_cast<double>(out_res[1][i]) - out_res[2][i]));
      scale = std::max(scale, std::abs(static_cast<double>(out_res[2][i])));
    }
  }
  std::cout << "steered MoE combine: plain_rows_exact=" << plain_rows_exact
            << " adapter rows worst |delta| " << worst << " of " << scale
            << '\n';
  return plain_rows_exact && worst <= 1e-5 * scale;
}

// Every residual-stream writer an adapter projects at runtime: the steered
// output of y = W x must equal the folded weights' W' x, W' = W - a r r^T W
// (rows of an embedding table: E' = E - a (E r) r^T, on each HC stream), to
// FP32 accuracy against an FP64 reference. Rows alternate base / adapter 1
// (every writer, `direction` or a synthetic unit vector, alpha 1) / adapter 2
// (block outputs only, v1): base rows and adapter 2's rows of the extra
// writer classes keep their bits.
bool CheckSteerWriters(const std::vector<float>& direction) {
  namespace w = gufo::models::qwen38_flash_next::steer_writer;
  struct Writer {
    const char* name;
    std::uint32_t in;
    std::uint8_t writer;  ///< 0: main-layer block output (no writer mask)
    std::uint32_t streams;
  };
  const Writer writers[] = {
      {"attention o_proj", 6144, 0, 1},
      {"GDN out_proj", 6144, 0, 1},
      {"shared expert down_proj", 640, 0, 1},
      {"routed expert down_proj", 640, 0, 1},
      {"PLE value_proj", 2560, w::kPle, 1},
      {"MTP o_proj", 6144, w::kMtp, 1},
      {"MTP expert down_proj", 640, w::kMtp, 1},
      {"token embedding rows", 0, w::kEmbedding, 4},
      {"MTP embedding rows", 0, w::kEmbedding, 1},
  };
  auto r = UnitDirections();
  if (!direction.empty())
    std::copy(direction.begin(), direction.end(), r.begin());
  const std::vector<float> alphas = {1.0F, 0.5F};
  const std::vector<std::uint8_t> masks = {
      static_cast<std::uint8_t>(w::kBlocks | w::kEmbedding | w::kPle | w::kMtp),
      w::kBlocks};
  const auto rows = AdapterRows();
  HipBuffer<float> d_r(r.size()), d_alpha(alphas.size());
  HipBuffer<std::uint8_t> d_rows(rows.size()), d_masks(masks.size());
  Upload(&d_r, r);
  Upload(&d_alpha, alphas);
  Upload(&d_rows, rows);
  Upload(&d_masks, masks);
  bool ok = true;
  std::uint32_t seed = 0xF01DU;
  for (const auto& wr : writers) {
    const bool table = wr.in == 0;
    // Real-scale weights: ~0.02 entries, unit-scale inputs; an embedding row
    // is a table row itself.
    const auto weights =
        table ? MakeValues(std::size_t{kTokens} * kHidden, ++seed, 0.05F)
              : MakeValues(std::size_t{kHidden} * wr.in, ++seed, 0.04F);
    const auto x = table
                       ? std::vector<float>{}
                       : MakeValues(std::size_t{kTokens} * wr.in, ++seed, 1.0F);
    // y = W x (float rows, as the GEMM writes them) and the folded W' x.
    std::vector<float> y(std::size_t{kTokens} * kHidden * wr.streams);
    std::vector<double> folded(std::size_t{kTokens} * kHidden);
    const std::vector<double> r64(r.begin(), r.begin() + kHidden);
    const double* rd = r64.data();
    std::vector<double> rw(table ? 0 : wr.in, 0.0);  // r^T W
    for (std::uint32_t o = 0; !table && o < kHidden; ++o)
      for (std::uint32_t k = 0; k < wr.in; ++k)
        rw[k] += rd[o] * weights[std::size_t{o} * wr.in + k];
    for (std::uint32_t t = 0; t < kTokens; ++t) {
      std::vector<double> plain(kHidden);
      double proj = 0.0;  // (r^T W) x, or e . r for a table row
      if (table) {
        for (std::uint32_t o = 0; o < kHidden; ++o) {
          plain[o] = weights[std::size_t{t} * kHidden + o];
          proj += rd[o] * plain[o];
        }
      } else {
        const float* xt = x.data() + std::size_t{t} * wr.in;
        for (std::uint32_t o = 0; o < kHidden; ++o) {
          const float* wo = weights.data() + std::size_t{o} * wr.in;
          double acc = 0.0;
          for (std::uint32_t k = 0; k < wr.in; ++k)
            acc += static_cast<double>(wo[k]) * xt[k];
          plain[o] = acc;
        }
        for (std::uint32_t k = 0; k < wr.in; ++k)
          proj += rw[k] * xt[k];
      }
      for (std::uint32_t o = 0; o < kHidden; ++o) {
        for (std::uint32_t s = 0; s < wr.streams; ++s)
          y[(std::size_t{t} * wr.streams + s) * kHidden + o] =
              static_cast<float>(plain[o]);
        folded[std::size_t{t} * kHidden + o] =
            plain[o] - alphas[0] * proj * rd[o];
      }
    }
    HipBuffer<float> d_y(y.size());
    Upload(&d_y, y);
    q::SteerRows steer{d_rows.get(), d_r.get(), d_alpha.get()};
    if (wr.writer != 0) {
      steer.writers = d_masks.get();
      steer.writer = wr.writer;
    }
    q::Steer(d_y.get(), steer, kTokens, kHidden, nullptr, wr.streams);
    CheckHip(hipDeviceSynchronize(), "writer steer synchronization");
    const auto got = Download(&d_y, y.size());
    double worst = 0.0, scale = 0.0;
    bool untouched = true;
    for (std::uint32_t t = 0; t < kTokens; ++t) {
      for (std::uint32_t s = 0; s < wr.streams; ++s) {
        const std::size_t row = (std::size_t{t} * wr.streams + s) * kHidden;
        if (rows[t] == 0 || (rows[t] == 2 && wr.writer != 0)) {
          untouched &= std::memcmp(y.data() + row, got.data() + row,
                                   kHidden * sizeof(float)) == 0;
          continue;
        }
        if (rows[t] != 1)
          continue;  // adapter 2's block outputs: CheckSteer covers alpha 0.5
        for (std::uint32_t o = 0; o < kHidden; ++o) {
          const double want = folded[std::size_t{t} * kHidden + o];
          worst = std::max(worst, std::abs(want - got[row + o]));
          scale = std::max(scale, std::abs(want));
        }
      }
    }
    const double relative = worst / scale;
    std::cout << "fold " << wr.name << ": max |steered - W'x| / max |W'x| "
              << relative << " untouched_rows_exact=" << untouched << '\n';
    ok &= untouched && relative < 1e-5;
  }
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    constexpr std::size_t kExpert =
        static_cast<std::size_t>(kTokens) * kSlots * kHidden;
    constexpr std::size_t kOut = static_cast<std::size_t>(kTokens) * kHidden;
    const auto expert_out = MakeValues(kExpert, 0x1234ABCDU, 1.0F);
    const auto weights = MakeValues(static_cast<std::size_t>(kTokens) * kSlots,
                                    0xBADC0FFEU, 0.5F, 0.5F);
    const auto shared = MakeValues(kOut, 0xDEADBEEFU, 1.0F);
    const auto gate = MakeValues(
        static_cast<std::size_t>(kTokens) * kGateStride, 0xC0FFEE11U, 3.0F);
    HipBuffer<float> d_expert(kExpert);
    HipBuffer<float> d_weights(weights.size());
    HipBuffer<float> d_shared(kOut);
    HipBuffer<float> d_gate(gate.size());
    HipBuffer<float> d_ref(kOut);
    HipBuffer<float> d_vec(kOut);
    Upload(&d_expert, expert_out);
    Upload(&d_weights, weights);
    Upload(&d_shared, shared);
    Upload(&d_gate, gate);
    q::MoeEpilogue(d_expert.get(), d_weights.get(), d_shared.get(),
                   d_gate.get(), kGateStride, d_ref.get(), kTokens, kSlots,
                   kHidden, nullptr);
    q::MoeEpilogueVec4(d_expert.get(), d_weights.get(), d_shared.get(),
                       d_gate.get(), kGateStride, d_vec.get(), kTokens, kSlots,
                       kHidden, nullptr);
    CheckHip(hipDeviceSynchronize(), "MoE epilogue synchronization");
    const auto ref = Download(&d_ref, kOut);
    const auto vec = Download(&d_vec, kOut);
    double worst = 0.0;
    for (std::size_t i = 0; i < kOut; ++i) {
      worst = std::max(worst, std::abs(static_cast<double>(ref[i] - vec[i])));
    }
    std::cout << "MoE epilogue vec4 worst absolute error " << worst << '\n';
    const bool steer = CheckSteer(false) && CheckSteer(true);
    const bool combine = CheckSteeredMoeCombine();
    // Optional: a real adapter direction (2,560 little-endian float32).
    std::vector<float> direction;
    if (argc > 1) {
      direction.resize(kHidden);
      std::ifstream in(argv[1], std::ios::binary);
      if (!in.read(reinterpret_cast<char*>(direction.data()),
                   kHidden * sizeof(float)))
        throw std::runtime_error(std::string("cannot read direction ") +
                                 argv[1]);
    }
    const bool writers = CheckSteerWriters(direction);
    return worst == 0.0 && steer && combine && writers ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

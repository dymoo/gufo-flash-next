// Routed expert decode kernels for the IQ formats of the smaller Unsloth
// quants (UD-IQ4_XS: IQ3_S gate/up, UD-Q3_K_XL: IQ3_XXS gate/up, both IQ4_XS
// on one layer and IQ4_NL down), reading the packed layout of iq.hpp.
// Included by moe.hip.cpp inside its anonymous namespace: same plan, work
// items, LDS staging and fixed FP32 order as the K-quant kernels there.
// Codebook values are exact int8 codes with one FP32 scale per 32 values
// (the Q8_0 arithmetic), so results are independent of batch width.

/// One work-group permutes one 32-row group from GGUF into iq.hpp's packed
/// layout through LDS (per byte: IQ tails are not dword multiples).
__global__ __launch_bounds__(256) void PackIqKernel(std::uint8_t* data,
                                                    int row_bytes,
                                                    WeightType type) {
  extern __shared__ std::uint32_t iq_tile_words[];
  const auto* src = reinterpret_cast<const std::uint8_t*>(iq_tile_words);
  auto* tile = data + std::size_t(blockIdx.x) * 32 * row_bytes;
  const int bytes = 32 * row_bytes;
  for (int i = threadIdx.x; i < bytes / 4; i += 256)
    iq_tile_words[i] = reinterpret_cast<const std::uint32_t*>(tile)[i];
  __syncthreads();
  const iq::Step st = iq::StepOf(type);
  for (int o = threadIdx.x; o < bytes; o += 256) {
    const int c = o / (32 * st.bytes);
    const int rem = o % (32 * st.bytes);
    int lane, b;
    if (rem < 512 * st.pieces) {
      lane = rem % 512 / 16;
      b = rem / 512 * 16 + rem % 16;
    } else {
      lane = (rem - 512 * st.pieces) / st.tail;
      b = 16 * st.pieces + (rem - 512 * st.pieces) % st.tail;
    }
    tile[o] = src[lane * row_bytes + c * st.bytes + iq::RawByte(type, b)];
  }
}

/// GateUpKernel for IQ gate/up rows (same work split and outputs). The
/// wave's 32 lanes are one packed 32-row group; per step (superblock) a lane
/// holds its canonical bytes in `w`: pieces, then the tail.
template<WeightType F>
__global__ __launch_bounds__(128) void GateUpIqKernel(
    const std::uint8_t* __restrict__ gate, const std::uint8_t* __restrict__ up,
    const Q8Block* __restrict__ xq, const int* __restrict__ entries,
    const int4* __restrict__ groups, const int* __restrict__ n_groups,
    Q8Block* __restrict__ hq, int used) {
  constexpr iq::Step kSt = iq::StepOf(F);
  constexpr int kP = kSt.pieces;
  constexpr int kWords = 4 * kP + (kSt.tail + 3) / 4;
  constexpr int kGrid = F == WeightType::kIQ3_S     ? 512
                        : F == WeightType::kIQ3_XXS ? 256
                                                    : 1;
  __shared__ Act act[2][kGroup][8];
  __shared__ float2 opr[2][kGroup][8];
  __shared__ float gate_out[kGateRows][kGroup + 1];
  __shared__ unsigned grid[kGrid];
  if (static_cast<int>(blockIdx.y) >= *n_groups)
    return;
  MOE_TRACE_BEGIN(0);
  const int4 g = groups[blockIdx.y];
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const bool is_up = wave >= 2;
  const int local = (wave & 1) * 32 + lane;
  const std::uint8_t* base =
      (is_up ? up : gate) +
      (std::size_t(g.x) * kFf + blockIdx.x * kGateRows + (wave & 1) * 32) *
          (kSt.bytes * kSuper);
  const int st = tid >> 3;
  const int sb = tid & 7;
  const bool stager = st < g.z;
  const Q8Block* src =
      xq + (stager ? std::size_t(entries[g.y + st] >> 4) * kXBlocks + sb : 0);
  if constexpr (F == WeightType::kIQ3_S) {
    for (int i = tid; i < 512; i += 128)
      grid[i] = iq3s_grid[i];
  } else if constexpr (F == WeightType::kIQ3_XXS) {
    for (int i = tid; i < 256; i += 128)
      grid[i] = iq3xxs_grid[i];
  }
  const auto load = [&](int s, unsigned* w) {
    const std::uint8_t* p = base + std::size_t(s) * 32 * kSt.bytes;
#pragma unroll
    for (int i = 0; i < kP; ++i) {
      const V4 v = reinterpret_cast<const V4*>(p)[i * 32 + lane];
      w[4 * i] = v.x;
      w[4 * i + 1] = v.y;
      w[4 * i + 2] = v.z;
      w[4 * i + 3] = v.w;
    }
    const std::uint8_t* t = p + 512 * kP + lane * kSt.tail;
    if constexpr (F == WeightType::kIQ3_S) {
      // 14-byte tails: odd lanes start 2 bytes into a dword.
      const int sh = (lane & 1) * 2;
      const auto* d = reinterpret_cast<const unsigned*>(t - sh);
      const unsigned a = d[0], b = d[1], c = d[2], e = d[3];
      w[4 * kP] = __builtin_amdgcn_alignbyte(b, a, sh);
      w[4 * kP + 1] = __builtin_amdgcn_alignbyte(c, b, sh);
      w[4 * kP + 2] = __builtin_amdgcn_alignbyte(e, c, sh);
      w[4 * kP + 3] = e >> (8 * sh);
    } else if constexpr (F == WeightType::kIQ3_XXS) {
      w[4 * kP] = *reinterpret_cast<const unsigned short*>(t);
    } else {
      const uint2 v = *reinterpret_cast<const uint2*>(t);
      w[4 * kP] = v.x;
      w[4 * kP + 1] = v.y;
    }
  };
  unsigned wa[kWords], wb[kWords];
  load(0, wa);
  if (stager)
    StoreAct(src[0], &act[0][st][sb], &opr[0][st][sb]);
  __syncthreads();
  float acc[kGroup] = {};
  const auto step = [&](int s, const unsigned(&w)[kWords], unsigned* next) {
    const int buf = s & 1;
    Q8Block xr;
    if (s + 1 < kSuper) {
      load(s + 1, next);
      if (stager)
        xr = src[(s + 1) * 8];
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float A;
      unsigned q[8];
      if constexpr (F == WeightType::kIQ3_S) {
        const float d = HalfLo(w[27]);
        A = iq::Iq3sScale(d, w[26], j);
        iq::Iq3sCodes(make_uint2(w[2 * j], w[2 * j + 1]),
                      iq::ByteOf(w[24 + j / 4], j % 4), w[16 + j], grid, q);
      } else if constexpr (F == WeightType::kIQ3_XXS) {
        A = iq::Iq3xxsScale(HalfLo(w[24]), w[16 + j]);
        iq::Iq3xxsCodes(make_uint2(w[2 * j], w[2 * j + 1]), w[16 + j], grid,
                        q);
      } else {
        A = iq::Iq4xsScale(w[32], w[33], j);
        const unsigned qs[4] = {w[4 * j], w[4 * j + 1], w[4 * j + 2],
                                w[4 * j + 3]};
        iq::Iq4Codes(qs, q);
      }
#pragma unroll
      for (int t = 0; t < kGroup; ++t) {
        if (t >= g.z)
          break;
        const Act x = act[buf][t][j];
        const int dot = Dot8(reinterpret_cast<const int*>(q), x.q[0], x.q[1]);
        acc[t] = __fmaf_rn(A, __fmul_rn(opr[buf][t][j].x, float(dot)), acc[t]);
      }
    }
    if (s + 1 < kSuper && stager)
      StoreAct(xr, &act[buf ^ 1][st][sb], &opr[buf ^ 1][st][sb]);
    __syncthreads();
  };
  static_assert(kSuper % 2 == 0);
  for (int s = 0; s < kSuper; s += 2) {
    step(s, wa, wb);
    step(s + 1, wb, wa);
  }
  if (!is_up) {
#pragma unroll
    for (int t = 0; t < kGroup; ++t)
      gate_out[local][t] = acc[t];
  }
  __syncthreads();
  if (is_up) {
    for (int t = 0; t < g.z; ++t) {
      const float gv = isfinite(gate_out[local][t]) ? gate_out[local][t] : 0.0f;
      const float uv = isfinite(acc[t]) ? acc[t] : 0.0f;
      const float e = __expf(-gv);
      const float h = __fmul_rn(
          __fmul_rn(gv, __builtin_amdgcn_rcpf(__fadd_rn(1.0f, e))), uv);
      const int entry = entries[g.y + t];
      QuantizeLane(
          h,
          hq + (std::size_t(entry >> 4) * used + (entry & 15)) * kHBlocks +
              blockIdx.x * 2 + (wave & 1),
          lane);
    }
  }
  MOE_TRACE_END(0);
}

/// DownKernel for packed IQ4_NL down rows: per two-block step a lane reads
/// its two code pieces and its d0 | d1 tail dword.
__global__ __launch_bounds__(128) void DownIq4NlKernel(
    const std::uint8_t* __restrict__ down, const Q8Block* __restrict__ hq,
    const int* __restrict__ entries, const int4* __restrict__ groups,
    const int* __restrict__ n_groups, float* __restrict__ out, int used) {
  constexpr int kStep = 36;
  constexpr int kSteps = kHBlocks / 2;
  constexpr int kAhead = 9;
  __shared__ Act act[kGroup][kHBlocks];
  __shared__ float2 opr[kGroup][kHBlocks];
  if (static_cast<int>(blockIdx.y) >= *n_groups)
    return;
  MOE_TRACE_BEGIN(1);
  const int4 g = groups[blockIdx.y];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int row = blockIdx.x * kDownRows + tid;
  const std::uint8_t* base =
      down + (std::size_t(g.x) * kHidden + row - lane) * (kStep * kSteps);
  unsigned wd[kSteps][9];
  const auto load = [&](int c) {
    const std::uint8_t* p = base + c * 32 * kStep;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const V4 v = reinterpret_cast<const V4*>(p)[i * 32 + lane];
      wd[c][4 * i] = v.x;
      wd[c][4 * i + 1] = v.y;
      wd[c][4 * i + 2] = v.z;
      wd[c][4 * i + 3] = v.w;
    }
    wd[c][8] = reinterpret_cast<const unsigned*>(p + 1024)[lane];
  };
#pragma unroll
  for (int c = 0; c < kAhead; ++c)
    load(c);
  for (int i = tid; i < g.z * kHBlocks; i += 128) {
    const int t = i / kHBlocks;
    const int b = i % kHBlocks;
    const int entry = entries[g.y + t];
    StoreAct(hq[(std::size_t(entry >> 4) * used + (entry & 15)) * kHBlocks + b],
             &act[t][b], &opr[t][b]);
  }
  __syncthreads();
  float acc[kGroup] = {};
#pragma unroll
  for (int c = 0; c < kSteps; ++c) {
    if (c + kAhead < kSteps)
      load(c + kAhead);
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const float scale = k ? HalfHi(wd[c][8]) : HalfLo(wd[c][8]);
      const unsigned qs[4] = {wd[c][4 * k], wd[c][4 * k + 1], wd[c][4 * k + 2],
                              wd[c][4 * k + 3]};
      unsigned q[8];
      iq::Iq4Codes(qs, q);
#pragma unroll
      for (int t = 0; t < kGroup; ++t) {
        if (t >= g.z)
          break;
        const Act x = act[t][2 * c + k];
        const int dot = Dot8(reinterpret_cast<const int*>(q), x.q[0], x.q[1]);
        acc[t] = __fmaf_rn(scale, __fmul_rn(opr[t][2 * c + k].x, float(dot)),
                           acc[t]);
      }
    }
  }
#pragma unroll
  for (int t = 0; t < kGroup; ++t) {
    if (t >= g.z)
      break;
    const int entry = entries[g.y + t];
    out[(std::size_t(entry >> 4) * used + (entry & 15)) * kHidden + row] =
        isfinite(acc[t]) ? acc[t] : 0.0f;
  }
  MOE_TRACE_END(1);
}

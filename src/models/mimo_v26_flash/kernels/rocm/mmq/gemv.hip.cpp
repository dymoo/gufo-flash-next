// SPDX-License-Identifier: MIT
// MiMo-V2.6-Flash decode-width projections: the vendored per-format dot
// products (vecdotq.hpp, the llama.cpp MMVQ arithmetic) in a grouped GEMV.
// A workgroup owns one output row of one matrix, reads it once and dots it
// with every activation row routed to that matrix (all rows for a dense
// projection, the expert's tokens for a routed one), NT rows per pass.
#include "mimo_mmq_prelude.h"
namespace mimo_mmq {
#include "mimo_mmq.h"
#include "common.hpp"
#include "vecdotq.hpp"
#include "quantize.hpp"

namespace {


template <ggml_type type>
struct Dot;
#define MIMO_DOT(T, fn, v)                                                    \
  template <>                                                               \
  struct Dot<T> {                                                           \
    static constexpr int vdr = v;                                           \
    static __device__ __forceinline__ float f(const void* x,                \
                                              const block_q8_1* y,          \
                                              const int& kbx,               \
                                              const int& iqs) {             \
      return fn(x, y, kbx, iqs);                                            \
    }                                                                       \
  }
MIMO_DOT(GGML_TYPE_Q6_K, vec_dot_q6_K_q8_1, VDR_Q6_K_Q8_1_MMVQ);
MIMO_DOT(GGML_TYPE_Q8_0, vec_dot_q8_0_q8_1, VDR_Q8_0_Q8_1_MMVQ);
MIMO_DOT(GGML_TYPE_Q3_K, vec_dot_q3_K_q8_1, VDR_Q3_K_Q8_1_MMVQ);
MIMO_DOT(GGML_TYPE_IQ2_XXS, vec_dot_iq2_xxs_q8_1, VDR_IQ2_XXS_Q8_1_MMVQ);
MIMO_DOT(GGML_TYPE_IQ2_XS, vec_dot_iq2_xs_q8_1, VDR_IQ2_XS_Q8_1_MMVQ);
MIMO_DOT(GGML_TYPE_IQ2_S, vec_dot_iq2_s_q8_1, VDR_IQ2_S_Q8_1_MMVQ);
MIMO_DOT(GGML_TYPE_IQ3_XXS, vec_dot_iq3_xxs_q8_1, VDR_IQ3_XXS_Q8_1_MMVQ);
MIMO_DOT(GGML_TYPE_IQ3_S, vec_dot_iq3_s_q8_1, VDR_IQ3_S_Q8_1_MMVQ);
MIMO_DOT(GGML_TYPE_MXFP4, vec_dot_mxfp4_q8_1, VDR_MXFP4_Q8_1_MMVQ);
#undef MIMO_DOT

/// One output row per workgroup of THREADS lanes (sized to the row's work,
/// so no lane idles on short rows); blockIdx.x is the row. Dense (bounds ==
/// nullptr): one matrix, activation rows [0, n). Routed: blockIdx.y is a
/// routed (token, slot) row; the first row of each expert's run owns that
/// expert, so only active experts get workgroups. NT activation rows share
/// each weight read; reductions skip slots a short run leaves empty.
template <ggml_type type, int NT, int THREADS>
__launch_bounds__(THREADS) __global__ void grouped_gemv(
    const char* __restrict__ w, const std::int64_t matrix_bytes,
    const block_q8_1* __restrict__ y, const int y_stride_blocks,
    const std::int32_t* __restrict__ ids_src1,
    const std::int32_t* __restrict__ ids_dst,
    const std::int32_t* __restrict__ bounds, const int n_dense,
    const int n_experts, float* __restrict__ out, const int M, const int K) {
  constexpr int qk = ggml_hip_type_traits<type>::qk;
  constexpr int qi = ggml_hip_type_traits<type>::qi;
  constexpr int vdr = Dot<type>::vdr;
  constexpr int per_block = qi / vdr;  // lanes sharing one quant block
  constexpr int blocks_per_iter = THREADS / per_block;
  constexpr int waves = THREADS / 32;

  int e = 0, lo = 0, hi = n_dense;
  if (bounds) {
    const int row = blockIdx.y;
    int a = 0, b = n_experts - 1;  // last expert with bounds[e] <= row
    while (a < b) {
      const int mid = (a + b + 1) / 2;
      if (bounds[mid] <= row)
        a = mid;
      else
        b = mid - 1;
    }
    e = a;
    lo = bounds[e];
    hi = bounds[e + 1];
    if (lo != row)
      return;
  }
  const int row = blockIdx.x;
  const int blocks_per_row = K / qk;
  const char* matrix = w + e * matrix_bytes;
  const int tid = threadIdx.x;
  const int kqs = vdr * (tid % per_block);
  __shared__ float partial[waves][NT];

  for (int base = lo; base < hi; base += NT) {
    const int nt = min(NT, hi - base);
    float acc[NT] = {};
    for (int kbx = tid / per_block; kbx < blocks_per_row;
         kbx += blocks_per_iter) {
      const int kby = kbx * (qk / QK8_1);
      const int xblock = row * blocks_per_row + kbx;
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        if (t >= nt)
          break;
        const int src = ids_src1 ? ids_src1[base + t] : base + t;
        acc[t] += Dot<type>::f(
            matrix, y + static_cast<std::int64_t>(src) * y_stride_blocks + kby,
            xblock, kqs);
      }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      if (t >= nt)
        break;
      float v = acc[t];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor(v, o, 32);
      if ((tid & 31) == 0)
        partial[tid >> 5][t] = v;
    }
    if constexpr (waves > 1)
      __syncthreads();
    if (tid < nt) {
      float v = 0.0f;
#pragma unroll
      for (int wv = 0; wv < waves; ++wv)
        v += partial[wv][tid];
      const int dst = ids_dst ? ids_dst[base + tid] : base + tid;
      out[static_cast<std::int64_t>(dst) * M + row] = v;
    }
    if constexpr (waves > 1)
      __syncthreads();
  }
}

template <ggml_type type, int NT, int THREADS>
void launch_shape(const void* W, std::int64_t matrix_bytes, const void* Y,
                  int y_stride, const std::int32_t* ids_src1,
                  const std::int32_t* ids_dst, const std::int32_t* bounds,
                  int n_dense, int grid_y, int n_experts, float* out, int M,
                  int K, hipStream_t stream) {
  grouped_gemv<type, NT, THREADS><<<dim3(M, grid_y), THREADS, 0, stream>>>(
      static_cast<const char*>(W), matrix_bytes,
      static_cast<const block_q8_1*>(Y), y_stride, ids_src1, ids_dst, bounds,
      n_dense, n_experts, out, M, K);
}

/// `rows_hint` bounds the activation rows any matrix receives (all rows for
/// a dense projection; at most the token count for a routed one).
template <ggml_type type>
int launch(const void* W, std::int64_t matrix_bytes, const void* Y,
           const std::int32_t* ids_src1, const std::int32_t* ids_dst,
           const std::int32_t* bounds, int n_dense, int grid_y, int n_experts,
           float* out, int M, int K, int rows_hint, hipStream_t stream) {
  constexpr int qk = ggml_hip_type_traits<type>::qk;
  constexpr int per_block =
      ggml_hip_type_traits<type>::qi / Dot<type>::vdr;
  const int y_stride = GGML_PAD(K, MATRIX_ROW_PADDING) / QK8_1;
  const int lanes = K / qk * per_block;  // lanes one pass over a row needs
  const auto go = [&](auto nt) {
    constexpr int NT = decltype(nt)::value;
    if (lanes <= 32)
      launch_shape<type, NT, 32>(W, matrix_bytes, Y, y_stride, ids_src1,
                                 ids_dst, bounds, n_dense, grid_y, n_experts,
                                 out, M, K, stream);
    else if (lanes <= 64)
      launch_shape<type, NT, 64>(W, matrix_bytes, Y, y_stride, ids_src1,
                                 ids_dst, bounds, n_dense, grid_y, n_experts,
                                 out, M, K, stream);
    else
      launch_shape<type, NT, 128>(W, matrix_bytes, Y, y_stride, ids_src1,
                                  ids_dst, bounds, n_dense, grid_y, n_experts,
                                  out, M, K, stream);
  };
  if (rows_hint <= 1)
    go(std::integral_constant<int, 1>{});
  else if (rows_hint <= 2)
    go(std::integral_constant<int, 2>{});
  else if (rows_hint <= 4)
    go(std::integral_constant<int, 4>{});
  else
    go(std::integral_constant<int, 8>{});
  return hipGetLastError() == hipSuccess ? 0 : -2;
}

template <typename F>
int dispatch(int type, F&& f) {
  switch ((ggml_type)type) {
    case GGML_TYPE_Q6_K: return f(std::integral_constant<ggml_type, GGML_TYPE_Q6_K>{});
    case GGML_TYPE_Q8_0: return f(std::integral_constant<ggml_type, GGML_TYPE_Q8_0>{});
    case GGML_TYPE_Q3_K: return f(std::integral_constant<ggml_type, GGML_TYPE_Q3_K>{});
    case GGML_TYPE_IQ2_XXS: return f(std::integral_constant<ggml_type, GGML_TYPE_IQ2_XXS>{});
    case GGML_TYPE_IQ2_XS: return f(std::integral_constant<ggml_type, GGML_TYPE_IQ2_XS>{});
    case GGML_TYPE_IQ2_S: return f(std::integral_constant<ggml_type, GGML_TYPE_IQ2_S>{});
    case GGML_TYPE_IQ3_XXS: return f(std::integral_constant<ggml_type, GGML_TYPE_IQ3_XXS>{});
    case GGML_TYPE_IQ3_S: return f(std::integral_constant<ggml_type, GGML_TYPE_IQ3_S>{});
    case GGML_TYPE_MXFP4: return f(std::integral_constant<ggml_type, GGML_TYPE_MXFP4>{});
    default: return -1;
  }
}

std::int64_t RowBytes(int type, int K) {
  return (std::int64_t)K / ggml_blck_size((ggml_type)type) *
         ggml_type_size((ggml_type)type);
}

}  // namespace

extern "C" size_t mimo_q8_1_bytes(int rows, int K) {
  return (size_t)rows * GGML_PAD(K, MATRIX_ROW_PADDING) / QK8_1 *
         sizeof(block_q8_1);
}

extern "C" int mimo_q8_1_quantize(const float* X, void* Y, int rows, int K,
                                  hipStream_t stream) {
  const int64_t padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
  quantize_row_q8_1_hip(X, nullptr, Y, GGML_TYPE_Q8_0, K, K, (int64_t)K * rows,
                        (int64_t)K * rows, padded, rows, 1, 1, stream);
  return hipGetLastError() == hipSuccess ? 0 : -2;
}

extern "C" int mimo_gemv_dense(int type, const void* W, const void* Y,
                               float* out, int M, int n, int K,
                               hipStream_t stream) {
  return dispatch(type, [&](auto t) {
    return launch<decltype(t)::value>(W, 0, Y, nullptr, nullptr, nullptr, n,
                                      1, 1, out, M, K, n, stream);
  });
}

extern "C" int mimo_gemv_moe(int type, const void* W, const void* Y,
                             const int32_t* ids_src1, const int32_t* ids_dst,
                             const int32_t* bounds, float* out, int M, int K,
                             int n_experts, int rows, int rows_per_expert,
                             hipStream_t stream) {
  return dispatch(type, [&](auto t) {
    // An expert receives at most one row per token: rows / used tokens.
    return launch<decltype(t)::value>(W, RowBytes(type, K) * M, Y, ids_src1,
                                      ids_dst, bounds, 0, rows, n_experts, out,
                                      M, K, rows_per_expert, stream);
  });
}

}  // namespace mimo_mmq

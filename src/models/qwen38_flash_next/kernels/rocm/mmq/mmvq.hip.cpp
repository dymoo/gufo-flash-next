#include "qfn_mmq_prelude.h"
namespace qfn_mmq {
#include "mmvq.hpp"
#include "unary.hpp"
#include "vecdotq.hpp"

// The vendored MMVQ dot product of each weight format against Q8_1
// activations. A K-quant super-block spans eight activation blocks.
// Each returns acc + dot. Q8_0 keeps the upstream `acc += dot` the
// production kernels were qualified with; the K-quant dots accumulate
// through explicit FMAs (vecdotq.hpp), batch-invariant under fast-math.
template<ggml_type type>
struct VecDot;
#define QFN_VEC_DOT(T, fn, v, expr)                                        \
  template<>                                                               \
  struct VecDot<T> {                                                       \
    static constexpr int vdr = v;                                          \
    static __device__ __forceinline__ float f(const void* x,               \
                                              const block_q8_1* y,         \
                                              const int& kbx,              \
                                              const int& iqs, float acc) { \
      return expr;                                                         \
    }                                                                      \
  }
QFN_VEC_DOT(GGML_TYPE_Q8_0, vec_dot_q8_0_q8_1, VDR_Q8_0_Q8_1_MMVQ,
            acc + vec_dot_q8_0_q8_1(x, y, kbx, iqs));
QFN_VEC_DOT(GGML_TYPE_Q4_K, vec_dot_q4_K_q8_1, VDR_Q4_K_Q8_1_MMVQ,
            vec_dot_q4_K_q8_1(x, y, kbx, iqs, acc));
QFN_VEC_DOT(GGML_TYPE_Q5_K, vec_dot_q5_K_q8_1, VDR_Q5_K_Q8_1_MMVQ,
            vec_dot_q5_K_q8_1(x, y, kbx, iqs, acc));
QFN_VEC_DOT(GGML_TYPE_Q6_K, vec_dot_q6_K_q8_1, VDR_Q6_K_Q8_1_MMVQ,
            vec_dot_q6_K_q8_1(x, y, kbx, iqs, acc));
#undef QFN_VEC_DOT

// Each wave handles up to eight dense inputs for one weight row on gfx1151.
template<int ncols_dst, bool has_gate, int token_waves = 1, bool ragged = false,
         ggml_type type = GGML_TYPE_Q8_0>
__launch_bounds__(32 * token_waves, 1) static __global__
    void mul_mat_vec_q8(const void* __restrict__ weights,
                        const void* __restrict__ gate,
                        const block_q8_1* __restrict__ input,
                        float* __restrict__ output, const uint32_t ncols_x,
                        const uint32_t nrows_x, const uint32_t stride_col_y,
                        const uint32_t valid_tokens = ncols_dst) {
  constexpr int qk = ggml_hip_type_traits<type>::qk;
  constexpr int qi = ggml_hip_type_traits<type>::qi;
  constexpr int vdr = VecDot<type>::vdr;
  constexpr int blocks_per_iter = vdr * 32 / qi;
  const int lane = token_waves > 1 ? threadIdx.x % 32 : threadIdx.x;
  const int first_token = token_waves > 1 ? threadIdx.x / 32 * ncols_dst : 0;
  const int row = blockIdx.x;
  const int blocks_per_row = ncols_x / qk;
  const int row_offset = row * blocks_per_row;
  const int kqs = vdr * (lane % (qi / vdr));
  float sum[ncols_dst] = {};
  float gate_sum[ncols_dst] = {};
  for (int kbx = lane / (qi / vdr); kbx < blocks_per_row;
       kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
      // The last wave may contain fewer than eight inputs. Its inactive
      // columns reuse a valid row; their results are never stored.
      const int token =
          ragged ? min(first_token + j, int(valid_tokens) - 1) : first_token + j;
      sum[j] = VecDot<type>::f(weights, input + token * stride_col_y + kby,
                               row_offset + kbx, kqs, sum[j]);
      if constexpr (has_gate) {
        gate_sum[j] =
            VecDot<type>::f(gate, input + j * stride_col_y + kby,
                            row_offset + kbx, kqs, gate_sum[j]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
    sum[j] = warp_reduce_sum<32>(sum[j]);
    if constexpr (has_gate)
      gate_sum[j] = warp_reduce_sum<32>(gate_sum[j]);
    if (lane == 0 && (token_waves == 1 || first_token + j < valid_tokens)) {
      float value = sum[j];
      if constexpr (has_gate)
        value *= ggml_hip_op_silu_single(gate_sum[j]);
      output[(first_token + j) * nrows_x + row] = value;
    }
  }
}

// Integer matrix products reuse each Q8 weight across up to 48 inputs.
// Keep four separate K8 sums per wave: merging them into a K32 integer sum
// would change the scalar kernel's rounded products, FMA chain and reduction.
template<int token_tiles, bool ragged>
__launch_bounds__(256) static __global__ void mul_mat_q8_decode_batch(
    const block_q8_0* __restrict__ weights,
    const block_q8_1* __restrict__ input, float* __restrict__ output,
    int k, int rows, int input_stride, int tokens) {
  using Int4 = __attribute__((ext_vector_type(4))) int;
  using Int8 = __attribute__((ext_vector_type(8))) int;
  __shared__ float partial[16 * 16 * 32];
  const int wave = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int sub = lane % 16;
  const int half = lane / 16;
  const int first_row = blockIdx.x * 16;
  const int blocks = k / QK8_0;
  float acc[token_tiles][4][8]{};
  for (int kb = wave; kb < blocks; kb += 8) {
    const auto* w = weights + min(first_row + sub, rows - 1) * blocks + kb;
    const float scale = __half2float(w->d);
    float scales[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      scales[i] = __shfl(scale, 2 * i + half, 32);
    Int4 codes[4];
#pragma unroll
    for (int part = 0; part < 4; ++part)
      codes[part] = {get_int_b2(w->qs, 2 * part),
                     get_int_b2(w->qs, 2 * part + 1), 0, 0};
#pragma unroll
    for (int tile = 0; tile < token_tiles; ++tile) {
      const int token = ragged && tile == token_tiles - 1
                            ? min(tile * 16 + sub, tokens - 1)
                            : tile * 16 + sub;
      const auto* x = input + token * input_stride + kb;
      const float x_scale = __low2float(x->ds);
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        const Int4 values = {get_int_b4(x->qs, 2 * part),
                              get_int_b4(x->qs, 2 * part + 1), 0, 0};
        Int8 dots{};
        dots = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true, codes[part], true, values, dots, false);
#pragma unroll
        for (int i = 0; i < 8; ++i)
          acc[tile][part][i] =
              __fmaf_rn(x_scale, __fmul_rn(scales[i], float(dots[i])),
                        acc[tile][part][i]);
      }
    }
  }
  // Reuse the 32 KiB transpose for each token tile. One thread reduces each
  // output in the original lane-zero tree, avoiding duplicate wave sums.
#pragma unroll
  for (int tile = 0; tile < token_tiles; ++tile) {
#pragma unroll
    for (int part = 0; part < 4; ++part) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int out = sub * 16 + 2 * i + half;
        const int swizzle = (out ^ (out >> 4)) & 31;
        partial[out * 32 + ((wave * 4 + part) ^ swizzle)] = acc[tile][part][i];
      }
    }
    __syncthreads();
    const int out = threadIdx.x;
    const int swizzle = (out ^ (out >> 4)) & 31;
    float sums[32];
#pragma unroll
    for (int i = 0; i < 32; ++i)
      sums[i] = partial[out * 32 + (i ^ swizzle)];
#pragma unroll
    for (int delta = 16; delta; delta /= 2) {
#pragma unroll
      for (int i = 0; i < delta; ++i)
        sums[i] = __fadd_rn(sums[i], sums[i + delta]);
    }
    const int row = first_row + out % 16;
    const int token = tile * 16 + out / 16;
    if (row < rows && (!ragged || token < tokens))
      output[token * rows + row] = sums[0];
    __syncthreads();
  }
}

template <int tokens, ggml_type type = GGML_TYPE_Q8_0>
static void launch_q8(const void* weights, const void* gate, const block_q8_1* input,
                      float* output, int k, int rows, int input_stride, hipStream_t stream) {
    if (gate) {
        mul_mat_vec_q8<tokens, true, 1, false, type><<<rows, 32, 0, stream>>>(
            weights, gate, input, output, k, rows, input_stride);
    } else {
        mul_mat_vec_q8<tokens, false, 1, false, type><<<rows, 32, 0, stream>>>(
            weights, nullptr, input, output, k, rows, input_stride);
    }
}

template <ggml_type type>
static void launch_vec(const void* weights, const void* gate,
                       const block_q8_1* input, float* output, int k, int rows,
                       int tokens, int input_stride, hipStream_t stream) {
  switch (tokens) {
    case 1:
      launch_q8<1, type>(weights, gate, input, output, k, rows, input_stride, stream);
      break;
    case 2:
      launch_q8<2, type>(weights, gate, input, output, k, rows, input_stride, stream);
      break;
    case 3:
      launch_q8<3, type>(weights, gate, input, output, k, rows, input_stride, stream);
      break;
    case 4:
      launch_q8<4, type>(weights, gate, input, output, k, rows, input_stride, stream);
      break;
    case 5:
      launch_q8<5, type>(weights, gate, input, output, k, rows, input_stride, stream);
      break;
    case 6:
      launch_q8<6, type>(weights, gate, input, output, k, rows, input_stride, stream);
      break;
    case 7:
      launch_q8<7, type>(weights, gate, input, output, k, rows, input_stride, stream);
      break;
    case 8:
      launch_q8<8, type>(weights, gate, input, output, k, rows, input_stride, stream);
      break;
    default:
      GGML_ABORT("invalid vector batch width");
  }
}

template<int waves>
static void launch_q8_batch(const void* weights, const block_q8_1* input,
                            float* output, int k, int rows, int tokens,
                            int input_stride, hipStream_t stream) {
  if (tokens % 8 != 0) {
    mul_mat_vec_q8<8, false, waves, true><<<rows, 32 * waves, 0, stream>>>(
        weights, nullptr, input, output, k, rows, input_stride, tokens);
  } else {
    mul_mat_vec_q8<8, false, waves><<<rows, 32 * waves, 0, stream>>>(
        weights, nullptr, input, output, k, rows, input_stride, tokens);
  }
}

template<int token_tiles>
static void launch_q8_matrix(const void* weights, const block_q8_1* input,
                             float* output, int k, int rows, int tokens,
                             int input_stride, hipStream_t stream) {
  if (tokens == 16 * token_tiles)
    mul_mat_q8_decode_batch<token_tiles, false>
        <<<(rows + 15) / 16, 256, 0, stream>>>(
            static_cast<const block_q8_0*>(weights), input, output, k, rows,
            input_stride, tokens);
  else
    mul_mat_q8_decode_batch<token_tiles, true>
        <<<(rows + 15) / 16, 256, 0, stream>>>(
            static_cast<const block_q8_0*>(weights), input, output, k, rows,
            input_stride, tokens);
}

void mul_mat_vec_q8_dispatch(const void* weights, const void* gate,
                             const block_q8_1* input, float* output, int k,
                             int rows, int tokens, int input_stride,
                             hipStream_t stream) {
  if (!gate && k == 2560 && rows >= 2560 && tokens >= 9 && tokens <= 48) {
    if (tokens <= 16)
      launch_q8_matrix<1>(weights, input, output, k, rows, tokens, input_stride,
                           stream);
    else if (tokens <= 32)
      launch_q8_matrix<2>(weights, input, output, k, rows, tokens, input_stride,
                           stream);
    else
      launch_q8_matrix<3>(weights, input, output, k, rows, tokens, input_stride,
                           stream);
    return;
  }
  GGML_ASSERT(k % QK8_0 == 0 && rows > 0);
  if (tokens > 8) {
    GGML_ASSERT(tokens <= 32 && !gate);
    // Each wave retains the eight-row arithmetic. Adjacent waves work
    // on the same weight row, reusing cache lines across requests.
    if (tokens <= 16) {
      launch_q8_batch<2>(weights, input, output, k, rows, tokens, input_stride,
                          stream);
    } else if (tokens <= 24) {
      launch_q8_batch<3>(weights, input, output, k, rows, tokens, input_stride,
                          stream);
    } else {
      launch_q8_batch<4>(weights, input, output, k, rows, tokens, input_stride,
                          stream);
    }
    return;
  }
  launch_vec<GGML_TYPE_Q8_0>(weights, gate, input, output, k, rows, tokens,
                             input_stride, stream);
}

void mul_mat_vec_k_dispatch(ggml_type type, const void* weights,
                            const void* gate, const block_q8_1* input,
                            float* output, int k, int rows, int tokens,
                            int input_stride, hipStream_t stream) {
  GGML_ASSERT(k % QK_K == 0 && rows > 0);
  // Wider batches run as consecutive groups of eight rows.
  for (int first = 0; first < tokens; first += MMVQ_MAX_BATCH_SIZE) {
    const int n = std::min(MMVQ_MAX_BATCH_SIZE, tokens - first);
    const block_q8_1* x = input + size_t(first) * input_stride;
    float* y = output + size_t(first) * rows;
    switch (type) {
      case GGML_TYPE_Q4_K:
        launch_vec<GGML_TYPE_Q4_K>(weights, gate, x, y, k, rows, n,
                                   input_stride, stream);
        break;
      case GGML_TYPE_Q5_K:
        launch_vec<GGML_TYPE_Q5_K>(weights, gate, x, y, k, rows, n,
                                   input_stride, stream);
        break;
      case GGML_TYPE_Q6_K:
        launch_vec<GGML_TYPE_Q6_K>(weights, gate, x, y, k, rows, n,
                                   input_stride, stream);
        break;
      default:
        GGML_ABORT("unsupported K-quant vector format");
    }
  }
}

}  // namespace qfn_mmq

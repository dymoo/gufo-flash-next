// SPDX-License-Identifier: MIT
// MiMo-V2.6-Flash adapters over the llama.cpp-derived quantized GEMMs (see
// VENDOR.md): the tiled MMQ path for every projection format this artifact
// uses. Decode-width shapes go through the model's own GEMV kernels.
#include "mimo_mmq_prelude.h"
namespace mimo_mmq {
#include "mimo_mmq.h"
#include "common.hpp"
#include "mmq.hpp"
#include "quantize.hpp"
#include "mmid.hpp"

extern "C" int mimo_mmq_init(int device) {
  if (device < 0)
    return -1;
  ggml_hip_set_device(device);
  const auto& info = ggml_hip_info();
  return device < info.device_count ? 0 : -1;
}

static ggml_backend_hip_context* context() {
  static std::unique_ptr<ggml_backend_hip_context> ctx;
  if (!ctx)
    ctx = std::make_unique<ggml_backend_hip_context>(ggml_hip_get_device());
  return ctx.get();
}

template <ggml_type type>
static int dense(const void* W, const float* X, float* out, int M, int N,
                 int K, hipStream_t stream) {
  const int cc = ggml_hip_info().devices[ggml_hip_get_device()].cc;
  ggml_backend_hip_context* ctx = context();
  const int64_t K_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
  const size_t y_bytes = (size_t)N * K_padded * sizeof(block_q8_1_mmq) /
                             (4 * QK8_1) +
                         get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
  ggml_hip_pool_alloc<char> y(ctx->pool(), y_bytes);
  quantize_mmq_q8_1_hip(X, nullptr, y.get(), type, K, K, 0, 0, K_padded, N,
                        1, 1, stream);
  const int64_t s01 = K / ggml_blck_size(type);
  const int64_t s12 =
      (int64_t)N * K_padded * sizeof(block_q8_1_mmq) / (4 * QK8_1 * sizeof(int));
  mmq_args args = {
      (const char*)W, type, (const int*)y.get(), nullptr, nullptr, out,
      K, M, N,
      s01, N, M,
      1, 1,
      0, s12, 0,
      1, 1,
      0, s12, 0,
      false,
      N,
  };
  mul_mat_q_case<type>(*ctx, args, stream);
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

template <ggml_type type>
static int moe(const void* W, const float* X, const int32_t* ids, float* out,
               int M, int K, int n_tokens, int n_experts, int used, int x_rows,
               hipStream_t stream) {
  const int cc = ggml_hip_info().devices[ggml_hip_get_device()].cc;
  ggml_backend_hip_context* ctx = context();
  const int64_t rows = (int64_t)n_tokens * used;
  ggml_hip_pool_alloc<int32_t> ids_src1(ctx->pool(), rows);
  ggml_hip_pool_alloc<int32_t> ids_dst(ctx->pool(), rows);
  ggml_hip_pool_alloc<int32_t> bounds(ctx->pool(), n_experts + 1);
  // x_rows == used gives each slot its own input row (it * used + slot).
  ggml_hip_launch_mm_ids_helper(ids, ids_src1.get(), ids_dst.get(),
                                bounds.get(), n_experts, n_tokens, used,
                                x_rows, used, x_rows, stream);
  const int64_t K_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
  const size_t y_bytes = rows * K_padded * sizeof(block_q8_1_mmq) /
                             (4 * QK8_1) +
                         get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
  ggml_hip_pool_alloc<char> y(ctx->pool(), y_bytes);
  quantize_mmq_q8_1_hip(X, ids_src1.get(), y.get(), type, K, K,
                        (int64_t)K * x_rows, (int64_t)K * x_rows * n_tokens,
                        K_padded, rows, 1, 1, stream);
  const int64_t s01 = K / ggml_blck_size(type);
  const int64_t s12 =
      K_padded * sizeof(block_q8_1_mmq) / (4 * QK8_1 * sizeof(int));
  mmq_args args = {
      (const char*)W, type, (const int*)y.get(), ids_dst.get(), bounds.get(),
      out,
      K, M, rows,
      s01, rows, M,
      n_experts, n_experts,
      (int64_t)M * s01, s12, 0,
      1, 1,
      0, s12 * n_tokens, 0,
      false,
      rows,
  };
  // No expert receives more rows than there are tokens: size the column-tile
  // grid by that bound (ncols_max still picks the tile width).
  args.ncols_grid_max = n_tokens;
  mul_mat_q_case<type>(*ctx, args, stream);
  return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int mimo_mmq_dense(int type, const void* W, const float* X,
                              float* out, int M, int N, int K,
                              hipStream_t stream) {
  switch ((ggml_type)type) {
    case GGML_TYPE_Q6_K:
      return dense<GGML_TYPE_Q6_K>(W, X, out, M, N, K, stream);
    case GGML_TYPE_Q8_0:
      return dense<GGML_TYPE_Q8_0>(W, X, out, M, N, K, stream);
    default:
      return -1;
  }
}

extern "C" int mimo_mmq_moe(int type, const void* W, const float* X,
                            const int32_t* ids, float* out, int M, int K,
                            int n_tokens, int n_experts, int used, int x_rows,
                            hipStream_t stream) {
#define MIMO_MOE(T) \
  case T:           \
    return moe<T>(W, X, ids, out, M, K, n_tokens, n_experts, used, x_rows, stream)
  switch ((ggml_type)type) {
    MIMO_MOE(GGML_TYPE_IQ2_XXS);
    MIMO_MOE(GGML_TYPE_IQ2_XS);
    MIMO_MOE(GGML_TYPE_IQ2_S);
    MIMO_MOE(GGML_TYPE_IQ3_XXS);
    MIMO_MOE(GGML_TYPE_IQ3_S);
    MIMO_MOE(GGML_TYPE_Q3_K);
    MIMO_MOE(GGML_TYPE_MXFP4);
    default:
      return -1;
  }
#undef MIMO_MOE
}

}  // namespace mimo_mmq

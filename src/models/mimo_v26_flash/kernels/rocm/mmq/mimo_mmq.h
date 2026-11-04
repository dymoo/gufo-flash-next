// SPDX-License-Identifier: MIT
#pragma once
#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>
extern "C" {

int mimo_mmq_init(int device);

/// out[N x M] = X[N x K] (F32) * W[M x K]^T, W in GGML `type` (Q6_K, Q8_0).
int mimo_mmq_dense(int type, const void* W, const float* X, float* out, int M,
                   int N, int K, hipStream_t stream);

/// Routed experts. W stacks n_experts matrices [M x K]; ids[n_tokens x used]
/// selects them. X holds `x_rows` rows per token: 1 (every slot reads the
/// token's row: gate/up) or `used` (one row per slot: down).
/// out[n_tokens x used x M].
int mimo_mmq_moe(int type, const void* W, const float* X, const int32_t* ids,
                 float* out, int M, int K, int n_tokens, int n_experts,
                 int used, int x_rows, hipStream_t stream);

/// Expert id maps (llama.cpp mm_ids_helper contract): rows sorted by expert,
/// `bounds[e]..bounds[e+1]` per expert, `ids_dst[i] = token * used + slot`.
int mimo_mmq_build_ids_maps(const int32_t* ids, int32_t* ids_src1,
                            int32_t* ids_dst, int32_t* expert_bounds,
                            int n_experts, int n_tokens, int n_expert_used,
                            int nchannels_y, int si1, int sis1,
                            hipStream_t stream);

/// q8_1 activations for the decode GEMVs: `rows` rows, K padded to 512.
size_t mimo_q8_1_bytes(int rows, int K);
int mimo_q8_1_quantize(const float* X, void* Y, int rows, int K,
                       hipStream_t stream);
/// Decode-width dense projection: out[n x M] = W[M x K] . Y[n].
int mimo_gemv_dense(int type, const void* W, const void* Y, float* out, int M,
                    int n, int K, hipStream_t stream);
/// Grouped expert projection over id maps; each expert's weights are read
/// once for all of its rows. `rows` = n_tokens * used; out[rows x M];
/// `rows_per_expert` bounds any expert's rows (the token count).
int mimo_gemv_moe(int type, const void* W, const void* Y,
                  const int32_t* ids_src1, const int32_t* ids_dst,
                  const int32_t* bounds, float* out, int M, int K,
                  int n_experts, int rows, int rows_per_expert,
                  hipStream_t stream);

}

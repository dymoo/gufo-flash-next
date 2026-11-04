// SPDX-License-Identifier: MIT
#pragma once
#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>
extern "C" {
int qfn_mmq_init(int device);

void qfn_mmq_set_routed_max_expert_rows(int rows);

void qfn_mmq_set_routed_tile_cols(int cols);

int qfn_mmq_routed_tile_cols_for_counts(const unsigned int *counts,
                                        int n_experts);

int qfn_mmq_q8_0_dense(
    const void  * W_q8_0,
    const float * X_f32,
    float       * out_f32,
    int           M,
    int           N,
    int           K,
    hipStream_t  stream);

size_t qfn_mmq_q8_1_bytes(int N, int K);

int qfn_mmq_quantize_q8_1(const float* X_f32, void* X_q8, int N, int K,
                          hipStream_t stream);

// Exact decode arithmetic for 1–8 rows, or up to 32 ungated rows.
// Wide projections (M >= 8192, K = 2560) accept up to 48 ungated rows.
// The 320×10240 HC projection also accepts larger multiples of eight.
// Only N output rows are written.
int qfn_mmq_q8_0_dense_vec_preq(const void* W_q8_0, const void* W_gate,
                                const void* X_q8, float* out_f32, int M, int N,
                                int K, hipStream_t stream);

// Q4_K, Q5_K and Q6_K dense projections; `type` is the GGUF type id and
// K % 256 == 0. The vector form takes Q8_1 rows (qfn_mmq_quantize_q8_1) for
// any N, in groups of eight; W_gate (same format and shape) makes it
// silu(gate) * W. The matrix form quantizes X itself (MMQ tiles).
int qfn_mmq_k_dense_vec_preq(int type, const void* W, const void* W_gate,
                             const void* X_q8, float* out_f32, int M, int N,
                             int K, hipStream_t stream);
int qfn_mmq_k_dense(int type, const void* W, const float* X_f32,
                    float* out_f32, int M, int N, int K, hipStream_t stream);

int qfn_mmq_build_ids_maps(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, hipStream_t stream);

int qfn_mmq_q8_0_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

int qfn_mmq_q4_K_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

int qfn_mmq_q5_1_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

int qfn_mmq_q5_K_moe_raw(
    const void * W, const float * X_f32, const int32_t * ids, float * out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

}

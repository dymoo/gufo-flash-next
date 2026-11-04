#pragma once
#include "common.hpp"

constexpr int MMVQ_MAX_BATCH_SIZE = 8;
void mul_mat_vec_q8_dispatch(const void* weights, const void* gate,
                            const block_q8_1* input, float* output,
                            int k, int rows, int tokens, int input_stride,
                            hipStream_t stream);
// Q4_K/Q5_K/Q6_K weights (k % 256 == 0), any number of activation rows.
void mul_mat_vec_k_dispatch(ggml_type type, const void* weights,
                            const void* gate, const block_q8_1* input,
                            float* output, int k, int rows, int tokens,
                            int input_stride, hipStream_t stream);

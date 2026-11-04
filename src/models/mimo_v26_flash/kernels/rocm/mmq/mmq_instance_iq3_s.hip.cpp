// SPDX-License-Identifier: MIT
// One MMQ instantiation per file so the formats compile in parallel.
#include "mimo_mmq_prelude.h"
namespace mimo_mmq {
#include "common.hpp"
#include "mmq.hpp"
template void mul_mat_q_case<GGML_TYPE_IQ3_S>(ggml_backend_hip_context& ctx,
                                              const mmq_args& args,
                                              hipStream_t stream);
}  // namespace mimo_mmq

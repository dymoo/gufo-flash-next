# Qwen3.8-Flash-Next (qwen4exp) model package. The scalar oracle is built
# only by explicit validation targets, not by production or routine checks.
add_library(gufo_qwen38_flash_next STATIC
  config.cpp
  ngram.cpp
  weights.cpp)

target_include_directories(gufo_qwen38_flash_next PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen38_flash_next PUBLIC gufo_core Threads::Threads)

add_library(gufo_qwen38_flash_next_reference STATIC EXCLUDE_FROM_ALL
  cpu_ops.cpp reference.cpp)
target_link_libraries(gufo_qwen38_flash_next_reference PUBLIC
  gufo_qwen38_flash_next)

# The reference decodes every weight on the fly; unoptimized it is unusable
# even as an oracle, so it keeps -O2 in every configuration.
set_source_files_properties(cpu_ops.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  set(QFN_ROCM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm")
  set(QFN_MMQ_DIR "${QFN_ROCM_DIR}/mmq")

  # HIP quantized matrix kernels adapted from llama.cpp, isolated in the
  # qfn_mmq namespace. The adapter retains its C++17 dialect.
  set(QFN_MMQ_SOURCES
    ${QFN_MMQ_DIR}/qfn_ggml_stubs.hip.cpp
    ${QFN_MMQ_DIR}/qfn_mmq.hip.cpp
    ${QFN_MMQ_DIR}/mmid.hip.cpp
    ${QFN_MMQ_DIR}/mmvq.hip.cpp
    ${QFN_MMQ_DIR}/quantize.hip.cpp)

  add_library(gufo_qwen38_flash_next_mmq STATIC ${QFN_MMQ_SOURCES})
  set_source_files_properties(${QFN_MMQ_SOURCES} PROPERTIES LANGUAGE HIP)
  set_target_properties(gufo_qwen38_flash_next_mmq PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(gufo_qwen38_flash_next_mmq PRIVATE
    GGML_HIP_NO_VMM)
  target_compile_options(gufo_qwen38_flash_next_mmq PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-value;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_qwen38_flash_next_mmq SYSTEM PUBLIC ${QFN_MMQ_DIR})
  target_link_libraries(gufo_qwen38_flash_next_mmq PUBLIC hip::host roc::hipblas)

  set(QFN_HIP_SOURCES
    ${QFN_ROCM_DIR}/kernels.hip.cpp
    ${QFN_ROCM_DIR}/moe.hip.cpp
    ${QFN_ROCM_DIR}/lfm.hip.cpp
    ${QFN_ROCM_DIR}/quants.hip.cpp
    ${QFN_ROCM_DIR}/spike.hip.cpp
    ${QFN_ROCM_DIR}/dense_i4.hip.cpp
    ${QFN_ROCM_DIR}/routed_iu4.hip.cpp
    ${QFN_ROCM_DIR}/routed_prefill.hip.cpp
    ${QFN_ROCM_DIR}/w8a8_wave64.hip
    ${QFN_ROCM_DIR}/blaslt.cpp
    ${QFN_ROCM_DIR}/device_model.cpp
    ${QFN_ROCM_DIR}/executor.cpp
    ${QFN_ROCM_DIR}/batch.cpp
    engine.cpp)
  target_sources(gufo_qwen38_flash_next PRIVATE ${QFN_HIP_SOURCES})
  set_source_files_properties(${QFN_ROCM_DIR}/kernels.hip.cpp
    ${QFN_ROCM_DIR}/moe.hip.cpp ${QFN_ROCM_DIR}/lfm.hip.cpp
    ${QFN_ROCM_DIR}/quants.hip.cpp ${QFN_ROCM_DIR}/spike.hip.cpp
    ${QFN_ROCM_DIR}/dense_i4.hip.cpp
    ${QFN_ROCM_DIR}/routed_iu4.hip.cpp ${QFN_ROCM_DIR}/routed_prefill.hip.cpp
    PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${QFN_ROCM_DIR}/w8a8_wave64.hip
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  target_compile_definitions(gufo_qwen38_flash_next PRIVATE QFN_ROCM_BUILD=1)
  target_compile_options(gufo_qwen38_flash_next PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_qwen38_flash_next SYSTEM PUBLIC
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR}
    ${ROCWMMA_INCLUDE_DIR})
  target_link_libraries(gufo_qwen38_flash_next PUBLIC
    gufo_qwen38_flash_next_mmq hip::host roc::hipblas roc::hipblaslt)

  # MiniCPM-V 4.7 images run through llama.cpp's mtmd. Point this at a
  # pinned llama.cpp tree whose HIP build (build-hip) has libmtmd and
  # libllama; without it, qwen35moe serves text only.
  set(GUFO_LLAMA_CPP_DIR "" CACHE PATH "llama.cpp tree with build-hip for mtmd")
  target_sources(gufo_qwen38_flash_next PRIVATE mtmd_vision.cpp)
  if (GUFO_LLAMA_CPP_DIR)
    set(GUFO_LLAMA_CPP_LIBS "${GUFO_LLAMA_CPP_DIR}/build-hip/bin")
    target_compile_definitions(gufo_qwen38_flash_next PUBLIC GUFO_WITH_MTMD=1)
    set_source_files_properties(mtmd_vision.cpp PROPERTIES INCLUDE_DIRECTORIES
      "${GUFO_LLAMA_CPP_DIR}/include;${GUFO_LLAMA_CPP_DIR}/ggml/include;${GUFO_LLAMA_CPP_DIR}/tools/mtmd")
    target_link_libraries(gufo_qwen38_flash_next PUBLIC
      "${GUFO_LLAMA_CPP_LIBS}/libmtmd.so" "${GUFO_LLAMA_CPP_LIBS}/libllama.so"
      "${GUFO_LLAMA_CPP_LIBS}/libggml.so" "${GUFO_LLAMA_CPP_LIBS}/libggml-base.so")
    target_link_options(gufo_qwen38_flash_next PUBLIC "-Wl,-rpath,${GUFO_LLAMA_CPP_LIBS}")
  endif()
endif()

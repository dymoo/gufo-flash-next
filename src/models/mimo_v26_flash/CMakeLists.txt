# MiMo-V2.6-Flash (mimo2) spike: gfx1151-only runtime specialised to the
# IQ2_M-class artifact's geometry and quant mix. HIP builds only.
set(MIMO_ROCM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm")
set(MIMO_MMQ_DIR "${MIMO_ROCM_DIR}/mmq")

# llama.cpp-derived quantized GEMMs in the mimo_mmq namespace (VENDOR.md).
set(MIMO_MMQ_SOURCES
  ${MIMO_MMQ_DIR}/mimo_ggml_stubs.hip.cpp
  ${MIMO_MMQ_DIR}/mimo_mmq.hip.cpp
  ${MIMO_MMQ_DIR}/gemv.hip.cpp
  ${MIMO_MMQ_DIR}/mmid.hip.cpp
  ${MIMO_MMQ_DIR}/quantize.hip.cpp)
foreach(format q6_k q8_0 q3_k iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s mxfp4)
  list(APPEND MIMO_MMQ_SOURCES ${MIMO_MMQ_DIR}/mmq_instance_${format}.hip.cpp)
endforeach()
add_library(gufo_mimo_v26_flash_mmq STATIC ${MIMO_MMQ_SOURCES})
set_source_files_properties(${MIMO_MMQ_SOURCES} PROPERTIES LANGUAGE HIP)
set_target_properties(gufo_mimo_v26_flash_mmq PROPERTIES
  HIP_STANDARD 17
  HIP_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(gufo_mimo_v26_flash_mmq PRIVATE GGML_HIP_NO_VMM)
target_compile_options(gufo_mimo_v26_flash_mmq PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-value;-Wno-unused-command-line-argument>)
target_include_directories(gufo_mimo_v26_flash_mmq SYSTEM PUBLIC ${MIMO_MMQ_DIR})
target_link_libraries(gufo_mimo_v26_flash_mmq PUBLIC hip::host roc::hipblas)

# chat_template.cpp builds into gufo_core: the chat protocol needs it on CPU.
add_library(gufo_mimo_v26_flash STATIC
  config.cpp
  weights.cpp
  engine.cpp
  ${MIMO_ROCM_DIR}/executor.cpp
  ${MIMO_ROCM_DIR}/kernels.hip.cpp)
set_source_files_properties(${MIMO_ROCM_DIR}/kernels.hip.cpp
  PROPERTIES LANGUAGE HIP)
target_include_directories(gufo_mimo_v26_flash PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(gufo_mimo_v26_flash PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O2>
  $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
target_link_libraries(gufo_mimo_v26_flash PUBLIC gufo_core
  gufo_mimo_v26_flash_mmq hip::host roc::hipblas)

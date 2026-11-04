# Small operator checks are the default. Full-model qualification and
# diagnostic probes are built explicitly so routine iterations stay focused.
add_custom_target(qwen38_flash_next_tests)

function(add_qwen38_flash_next_test name library labels)
  set(target qwen38_flash_next_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  target_link_libraries(${target} PRIVATE ${library})
  add_test(NAME qwen38_flash_next.${name} COMMAND ${target})
  set_tests_properties(qwen38_flash_next.${name} PROPERTIES
    LABELS "qwen38_flash_next;${labels}")
  add_dependencies(qwen38_flash_next_tests ${target})
endfunction()

add_qwen38_flash_next_test(config gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(ngram gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(weights gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(mtp_sampling gufo_core "cpu")
add_qwen38_flash_next_test(prompt_lookup gufo_core "cpu")
add_qwen38_flash_next_test(dspark_policy gufo_core "cpu")
add_qwen38_flash_next_test(spike_knobs gufo_core "cpu")

add_executable(qwen38_flash_next_reference_probe EXCLUDE_FROM_ALL reference_probe.cpp)
target_link_libraries(qwen38_flash_next_reference_probe PRIVATE
  gufo_qwen38_flash_next_reference)
set_source_files_properties(reference_probe.cpp PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  foreach(name IN ITEMS mtp_ops blaslt_ops hc_mix_ops gdn_ops
      attention_ops moe_epilogue_ops moe_ids_ops moe_ops moe_routing_ops
      projection_ops routed_wmma_ops select_ops lfm_ops)
    add_qwen38_flash_next_test(${name} gufo_qwen38_flash_next "hip;gpu;gfx1151")
  endforeach()

  add_executable(qwen38_flash_next_session_test EXCLUDE_FROM_ALL session_test.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_disk_store.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_model_runner.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_generation_scheduler.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/inference_backend.cpp)
  target_link_libraries(qwen38_flash_next_session_test PRIVATE gufo_logging
    gufo_structured gufo_qwen38_flash_next gufo_deepseek_v4_flash gufo_mimo_v26_flash hip::host)
  add_executable(qwen38_flash_next_snapshot_test EXCLUDE_FROM_ALL snapshot_test.cpp)
  target_link_libraries(qwen38_flash_next_snapshot_test PRIVATE
    gufo_qwen38_flash_next)
  add_custom_target(qwen38_flash_next_model_tests
    DEPENDS qwen38_flash_next_session_test qwen38_flash_next_snapshot_test)

  add_executable(qwen38_flash_next_moe_bench EXCLUDE_FROM_ALL moe_bench.cpp)
  set_source_files_properties(moe_bench.cpp PROPERTIES LANGUAGE HIP)
  target_link_libraries(qwen38_flash_next_moe_bench PRIVATE
    gufo_qwen38_flash_next)

  add_executable(qwen38_flash_next_batch_dense_bench EXCLUDE_FROM_ALL
    batch_dense_bench.cpp)
  target_link_libraries(qwen38_flash_next_batch_dense_bench PRIVATE
    gufo_qwen38_flash_next)

  add_executable(qwen38_flash_next_dense_i4_bench EXCLUDE_FROM_ALL
    dense_i4_bench.cpp)
  target_link_libraries(qwen38_flash_next_dense_i4_bench PRIVATE
    gufo_qwen38_flash_next)

  add_executable(qwen38_flash_next_spike_bench EXCLUDE_FROM_ALL
    spike_bench.cpp)
  target_link_libraries(qwen38_flash_next_spike_bench PRIVATE
    gufo_qwen38_flash_next)

  add_executable(qwen38_flash_next_prefill_bench EXCLUDE_FROM_ALL
    prefill_bench.cpp)
  target_link_libraries(qwen38_flash_next_prefill_bench PRIVATE
    gufo_qwen38_flash_next)

  add_executable(qwen38_flash_next_hc_chain_bench EXCLUDE_FROM_ALL
    hc_chain_bench.cpp)
  target_link_libraries(qwen38_flash_next_hc_chain_bench PRIVATE
    gufo_qwen38_flash_next)

  add_executable(qwen38_flash_next_dspark_probe EXCLUDE_FROM_ALL
    dspark_probe.cpp)
  target_link_libraries(qwen38_flash_next_dspark_probe PRIVATE
    gufo_qwen38_flash_next)

  add_executable(qwen38_flash_next_gpu_probe EXCLUDE_FROM_ALL gpu_probe.cpp
    mtp_audit.cpp)
  target_link_libraries(qwen38_flash_next_gpu_probe PRIVATE
    gufo_qwen38_flash_next_reference)

  # Frequency-ranked MTP draft vocabulary (GUFO_MTP_VOCAB) from corpora.
  add_executable(qwen38_flash_next_spec_vocab EXCLUDE_FROM_ALL spec_vocab.cpp)
  target_link_libraries(qwen38_flash_next_spec_vocab PRIVATE
    gufo_qwen38_flash_next)
endif()

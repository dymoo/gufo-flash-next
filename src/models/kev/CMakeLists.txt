# Kev-4B (TypeSafe System One) on the Qwen3.5 hybrid backbone.

target_sources(gufo_core PRIVATE
  ${PROJECT_SOURCE_DIR}/src/models/kev/systemone.cpp
  ${PROJECT_SOURCE_DIR}/src/models/kev/encoding.cpp
  ${PROJECT_SOURCE_DIR}/src/models/kev/model.cpp
  ${PROJECT_SOURCE_DIR}/src/models/kev/hip/packed.cpp)

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_core PRIVATE
    ${PROJECT_SOURCE_DIR}/src/models/kev/hip/kernels.hip)
endif()

function(gufo_kev_register_tests)
  add_executable(kev_systemone_test
    ${PROJECT_SOURCE_DIR}/src/models/kev/tests/systemone_test.cpp)
  target_link_libraries(kev_systemone_test PRIVATE gufo_core)
  add_test(NAME kev_systemone_test COMMAND kev_systemone_test)
  set_tests_properties(kev_systemone_test PROPERTIES
    LABELS "cpu;models;kev;systemone;quality")

  add_executable(kev_parity_test
    ${PROJECT_SOURCE_DIR}/src/models/kev/tests/parity_test.cpp)
  target_link_libraries(kev_parity_test PRIVATE gufo_core)
  add_test(NAME kev_parity_test COMMAND kev_parity_test)
  set_tests_properties(kev_parity_test PROPERTIES
    LABELS "gpu;models;kev;quality;external-model" SKIP_RETURN_CODE 77)

  add_executable(kev_bench ${PROJECT_SOURCE_DIR}/src/models/kev/tests/bench.cpp)
  target_link_libraries(kev_bench PRIVATE gufo_core)
endfunction()

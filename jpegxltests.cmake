# Copyright (c) the JPEG XL Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(TEST_FILES
  jxl/ac_strategy_test.cc
  jxl/adaptive_reconstruction_test.cc
  jxl/alpha_test.cc
  jxl/ans_common_test.cc
  jxl/ans_test.cc
  jxl/base/fast_log_test.cc
  jxl/bit_reader_test.cc
  jxl/bits_test.cc
  jxl/brotli_test.cc
  jxl/brunsli_test.cc
  jxl/byte_order_test.cc
  jxl/coeff_order_test.cc
  jxl/color_encoding_test.cc
  jxl/color_management_test.cc
  jxl/compressed_dc_test.cc
  jxl/compressed_image_test.cc
  jxl/convolve_test.cc
  jxl/data_parallel_test.cc
  jxl/dct_test.cc
  jxl/dct_util_test.cc
  jxl/descriptive_statistics_test.cc
  jxl/entropy_coder_test.cc
  jxl/external_image_test.cc
  jxl/extras/codec_test.cc
  jxl/fields_test.cc
  jxl/gaborish_test.cc
  jxl/gamma_correct_test.cc
  jxl/gradient_test.cc
  jxl/iaca_test.cc
  jxl/icc_codec_test.cc
  jxl/image_test.cc
  jxl/jpeg_kpasses_test.cc
  jxl/jpegxl/decode_test.cc
  jxl/jxl_test.cc
  jxl/lehmer_code_test.cc
  jxl/linalg_test.cc
  jxl/opsin_image_test.cc
  jxl/opsin_inverse_test.cc
  jxl/optimize_test.cc
  jxl/padded_bytes_test.cc
  jxl/passes_test.cc
  jxl/predictor_test.cc
  jxl/preview_test.cc
  jxl/quant_weights_test.cc
  jxl/quantizer_test.cc
  jxl/rational_polynomial_test.cc
  jxl/robust_statistics_test.cc
  jxl/speed_tier_test.cc
  jxl/splines_test.cc
  jxl/threads/thread_parallel_runner_test.cc
  jxl/toc_test.cc
  jxl/xorshift128plus_test.cc
  tools/box/box_test.cc
)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)
foreach (TESTFILE IN LISTS TEST_FILES)
  # The TESTNAME is the name without the extension or directory.
  get_filename_component(TESTNAME ${TESTFILE} NAME_WE)
  add_executable(${TESTNAME} ${TESTFILE})
  target_include_directories(${TESTNAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fastapprox
  )
  if(JPEGXL_EMSCRIPTEN)
    target_compile_definitions(${TESTNAME} PRIVATE
        -DTEST_DATA_PATH="/")
    # The emscripten linking step takes too much memory and crashes during the
    # wasm-opt step when using -O2 optimization level
    set_target_properties(${TESTNAME} PROPERTIES LINK_FLAGS "\
      -O1 \
      -s TOTAL_MEMORY=1536MB \
      --embed-file ${CMAKE_CURRENT_SOURCE_DIR}/third_party/testdata@/ \
    ")

  else()
    target_compile_definitions(${TESTNAME} PRIVATE
        -DTEST_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/third_party/testdata")
  endif()
  target_compile_options(${TESTNAME} PRIVATE
    ${JPEGXL_INTERNAL_FLAGS}
    # Add coverage flags to the test binary so code in the private headers of
    # the library is also instrumented when running tests that execute it.
    ${JPEGXL_COVERAGE_FLAGS}
  )
  target_link_libraries(${TESTNAME}
    box
    jpegxl-static
    jpegxl_threads
    jpegxl_extras-static
    gmock
    gtest
    gtest_main
  )
  # Output test targets in the test directory.
  set_target_properties(${TESTNAME} PROPERTIES PREFIX "tests/")
  if (WIN32 AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set_target_properties(${TESTNAME} PROPERTIES COMPILE_FLAGS "-Wno-error")
  endif ()
  if(${CMAKE_VERSION} VERSION_LESS "3.10.3")
    gtest_discover_tests(${TESTNAME} TIMEOUT 30)
  else ()
    gtest_discover_tests(${TESTNAME} DISCOVERY_TIMEOUT 30)
  endif ()
endforeach ()

# googletest doesn't compile clean with clang-cl (-Wundef)
if (WIN32 AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set_target_properties(gtest PROPERTIES COMPILE_FLAGS "-Wno-error")
  set_target_properties(gtest_main PROPERTIES COMPILE_FLAGS "-Wno-error")
endif ()
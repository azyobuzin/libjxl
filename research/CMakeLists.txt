# 依存ライブラリ
find_package(Boost REQUIRED COMPONENTS program_options serialization unit_test_framework)
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED core imgcodecs imgproc)
find_package(TBB REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(MLPACK REQUIRED mlpack>=3.0.0)

# mlpack が OpenMP を使う
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# 共通ライブラリ
add_library(research STATIC EXCLUDE_FROM_ALL
  cost_graph_enc.cc
  dec_jxl_multi.cc
  enc_brute_force.cc
  enc_jxl_multi.cc
  fields.cc
  images_provider.cc
  progress.cc
  prop_extract.cc
)
target_compile_options(research PUBLIC "${JPEGXL_INTERNAL_FLAGS}")
target_link_libraries(research jxl-static fmt::fmt ${OpenCV_LIBS} ${TBB_IMPORTED_TARGETS})
target_include_directories(research PUBLIC "${PROJECT_SOURCE_DIR}" PRIVATE edmonds-alg/src)

# Executables
add_executable(cost_graph_enc cost_graph_enc_main.cc)
target_link_libraries(cost_graph_enc research ${Boost_LIBRARIES})

add_executable(enc_brute_force enc_brute_force_main.cc)
target_link_libraries(enc_brute_force research ${Boost_LIBRARIES})

add_executable(enc_without_header enc_without_header_main.cc)
target_link_libraries(enc_without_header research ${Boost_LIBRARIES})

add_executable(prop_extract prop_extract_main.cc)
target_link_libraries(prop_extract research ${Boost_LIBRARIES})

add_executable(prop_kmeans prop_kmeans_main.cc)
target_link_libraries(prop_kmeans research ${Boost_LIBRARIES} ${MLPACK_LINK_LIBRARIES})

add_custom_target(research_exe ALL DEPENDS cost_graph_enc enc_brute_force enc_without_header prop_extract prop_kmeans)

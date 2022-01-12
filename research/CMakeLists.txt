# 依存ライブラリ
find_package(Boost REQUIRED COMPONENTS iostreams program_options serialization unit_test_framework)
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED core imgcodecs imgproc)
find_package(TBB REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(MLPACK REQUIRED mlpack>=3.0.0)
pkg_check_modules(GMP REQUIRED gmpxx)

# mlpack が OpenMP を使う
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# 共通ライブラリ
add_library(research STATIC EXCLUDE_FROM_ALL
  cocbo.cc
  cost_graph_enc.cc
  common_cluster.cc
  dec_cluster.cc
  dec_flif.cc
  enc_all.cc
  enc_brute_force.cc
  enc_cluster.cc
  enc_flif.cc
  images_provider.cc
  jxl_parallel.cc
  progress.cc
  prop_extract.cc
  flif/common.cpp
  flif/io.cpp
  flif/image/color_range.cpp
  flif/image/crc32k.cpp
  flif/maniac/bit.cpp
  flif/maniac/chance.cpp
  flif/maniac/symbol.cpp
  flif/transform/factory.cpp
)
target_compile_options(research PUBLIC "${JPEGXL_INTERNAL_FLAGS}")
target_link_libraries(research jxl-static fmt::fmt glpk ${GMP_LINK_LIBRARIES} ${MLPACK_LINK_LIBRARIES} ${OpenCV_LIBS} ${TBB_IMPORTED_TARGETS})
target_include_directories(research PUBLIC "${PROJECT_SOURCE_DIR}" PRIVATE edmonds-alg/src)

# Executables
add_executable(benchmark_random_decode benchmark_random_decode_main.cc)
target_link_libraries(benchmark_random_decode research ${Boost_LIBRARIES})

add_executable(cost_graph_enc cost_graph_enc_main.cc)
target_link_libraries(cost_graph_enc research ${Boost_LIBRARIES})

add_executable(enc_all enc_all_main.cc)
target_link_libraries(enc_all research ${Boost_LIBRARIES})

add_executable(enc_avg enc_avg_main.cc)
target_link_libraries(enc_avg research ${Boost_LIBRARIES})

add_executable(enc_brute_force enc_brute_force_main.cc)
target_link_libraries(enc_brute_force research ${Boost_LIBRARIES})

add_executable(enc_without_header enc_without_header_main.cc)
target_link_libraries(enc_without_header research ${Boost_LIBRARIES})

add_executable(prop_clustering prop_clustering_main.cc)
target_link_libraries(prop_clustering research ${Boost_LIBRARIES})

add_executable(prop_extract prop_extract_main.cc)
target_link_libraries(prop_extract research ${Boost_LIBRARIES})

add_executable(test_cluster_pointers test_cluster_pointers_main.cc)
target_link_libraries(test_cluster_pointers research)

add_custom_target(research_exe ALL DEPENDS
  benchmark_random_decode
  cost_graph_enc
  enc_all
  enc_avg
  enc_brute_force
  enc_without_header
  prop_clustering
  prop_extract
  test_cluster_pointers
)

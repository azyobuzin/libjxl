# 依存ライブラリ
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED core imgcodecs imgproc)
find_package(TBB REQUIRED)

# 共通ライブラリ
add_library(research STATIC EXCLUDE_FROM_ALL
  images_provider.cc
  prop_extract.cc
)
target_compile_options(research PUBLIC "${JPEGXL_INTERNAL_FLAGS}")
target_link_libraries(research jxl-static fmt::fmt ${OpenCV_LIBS} ${TBB_IMPORTED_TARGETS})
target_include_directories(research PUBLIC "${PROJECT_SOURCE_DIR}")

# Executables
add_executable(prop_extract prop_extract_main.cc)
target_link_libraries(prop_extract research ${Boost_LIBRARIES})

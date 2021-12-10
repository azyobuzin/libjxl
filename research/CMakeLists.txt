# 共通ライブラリ
add_library(research STATIC EXCLUDE_FROM_ALL
  prop_extract.cc
)
target_compile_options(research PUBLIC "${JPEGXL_INTERNAL_FLAGS}")
target_link_libraries(research jxl-static)
target_include_directories(research PUBLIC "${PROJECT_SOURCE_DIR}")

# Executables
add_executable(prop_extract prop_extract_main.cc)
target_link_libraries(prop_extract research)

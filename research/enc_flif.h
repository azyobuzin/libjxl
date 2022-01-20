#pragma once

#include <memory>
#include <opencv2/core.hpp>

#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/modular/modular_image.h"

namespace research {

jxl::PaddedBytes EncodeColorSignalWithFlif(
    const std::vector<std::shared_ptr<const jxl::Image>>& images,
    int learn_repeats, int additional_props);

size_t ComputeEncodedBytesWithFlif(const cv::Mat& rgb_image);

}  // namespace research

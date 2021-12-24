#pragma once

#include <vector>

#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/modular/modular_image.h"

namespace research {

jxl::Status DecodeColorSignalWithFlif(std::vector<jxl::Image>& images,
                                      jxl::Span<const uint8_t> data,
                                      int additional_props);

}

#pragma once

#include <memory>

#include "lib/jxl/modular/modular_image.h"

namespace research {

std::unique_ptr<uint8_t[]> EncodeColorSignalWithFlif(const jxl::Image& image);

}

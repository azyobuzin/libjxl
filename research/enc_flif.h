#pragma once

#include <memory>

#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/modular/modular_image.h"

namespace research {

jxl::PaddedBytes EncodeColorSignalWithFlif(
    const std::vector<jxl::Image>& images, int learn_repeats);

}

#include "enc_flif.h"
#include "lib/jxl/base/status.h"
#include "flif/flifio.hpp"
#include "flif/flif-enc.cpp"

using namespace jxl;

namespace research {

std::unique_ptr<uint8_t[]> EncodeColorSignalWithFlif(const jxl::Image& image) {
  JXL_CHECK(image.channel.size() - image.nb_meta_channels == 3);

  ::Image flif_image(image.w, image.h, 0, 255, 3);

  std::vector<PropertySymbolCoder<FLIFBitChancePass1, RacDummy, 10>> coders;
  coders.reserve(2);

  //TODO
}

}

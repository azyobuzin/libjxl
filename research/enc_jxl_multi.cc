#include "enc_jxl_multi.h"

namespace research {

void EncodeMultiImages(jxl::BitWriter &writer,
                       const std::vector<const jxl::Image &> &images,
                       const jxl::ModularOptions &options, size_t max_refs) {
  max_refs = std::min(max_refs, images.size() - 1);
  // TODO
}

}  // namespace research

#include <vector>

#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"

namespace research {

// 複数枚を JPEG XL で圧縮する。
// max_refs で前何枚までの画像を参照するかを指定する。
void EncodeMultiImages(jxl::BitWriter &writer,
                       const std::vector<std::shared_ptr<jxl::Image>> &images,
                       const jxl::ModularOptions &options, size_t max_refs);

}  // namespace research

#pragma once

#include <vector>

#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/modular/transform/transform.h"

namespace research {

// 複数の画像をまとめた画像
struct CombinedImage {
  jxl::Image image;
  size_t n_images;
  CombinedImage(jxl::Image image, size_t n_images);
};

struct EncodedCombinedImage {
  // data に含まれる画像のインデックス
  std::vector<size_t> image_indices;
  std::vector<std::shared_ptr<const jxl::Image>> included_images;
  jxl::PaddedBytes data;
  // 書き込まれたビット数
  size_t n_bits;
};

CombinedImage CombineImage(jxl::Image &&image);

CombinedImage CombineImage(
    const std::vector<std::shared_ptr<const jxl::Image>> &images);

jxl::Tree LearnTree(jxl::BitWriter &writer, const CombinedImage &image,
                    const jxl::ModularOptions &options, size_t max_refs);

// 複数枚を JPEG XL で圧縮する。
// max_refs で前何枚までの画像を参照するかを指定する。
void EncodeImages(jxl::BitWriter &writer, const CombinedImage &image,
                  const jxl::ModularOptions &options, size_t max_refs,
                  const jxl::Tree &tree);

}  // namespace research

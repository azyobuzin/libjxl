#pragma once

#include <ostream>
#include <vector>

#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/modular/transform/transform.h"

namespace research {

struct EncodingOptions {
  jxl::ParentReferenceType parent_reference;
  bool flif_enabled;
  int flif_learn_repeats;
  int flif_additional_props;
};

// 複数の画像をまとめた画像
struct CombinedImage {
  jxl::Image image;
  uint32_t n_images;
  // 2番目以降の画像が何番目を参照するか
  std::vector<uint32_t> references;
};

struct EncodedCombinedImage {
  // data に含まれる画像のインデックス
  std::vector<uint32_t> image_indices;
  std::vector<std::shared_ptr<const jxl::Image>> included_images;
  std::vector<uint32_t> references;
  jxl::PaddedBytes data;
  jxl::PaddedBytes flif_data;

  size_t n_bytes() const noexcept { return data.size() + flif_data.size(); }
};

int FindBestWPMode(const jxl::Image &image);

CombinedImage CombineImage(jxl::Image &&image);

CombinedImage CombineImage(
    const std::vector<std::shared_ptr<const jxl::Image>> &images,
    std::vector<uint32_t> references);

jxl::Tree LearnTree(jxl::BitWriter &writer, const CombinedImage &image,
                    jxl::ModularOptions &options,
                    jxl::ParentReferenceType parent_reference);

// 複数枚を JPEG XL で圧縮する。
void EncodeImages(jxl::BitWriter &writer, const CombinedImage &image,
                  const jxl::ModularOptions &options,
                  jxl::ParentReferenceType parent_reference,
                  const jxl::Tree &tree);

// 元のi番目の画像は、combined_imagesの何番目の画像かを表すpointersを符号化する
void EncodeClusterPointers(jxl::BitWriter &writer,
                           const std::vector<uint32_t> pointers);

// i-1番目の画像は、何番目の画像を参照するのかを表すreferencesを符号化する
void EncodeReferences(jxl::BitWriter &writer,
                      jxl::ParentReferenceType parent_reference,
                      const std::vector<uint32_t> references);

void PackToClusterFile(const std::vector<EncodedCombinedImage> &combined_images,
                       jxl::ParentReferenceType parent_reference,
                       std::ostream &dst);

}  // namespace research

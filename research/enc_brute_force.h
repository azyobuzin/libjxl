#pragma once

#include "cost_graph.h"
#include "enc_cluster.h"

namespace research {

// MSTから総当たりで、圧縮率が良くなるケースだけひとつの画像にまとめる
// TODO(research): Cost をテンプレートにしたい
std::vector<EncodedCombinedImage> EncodeWithBruteForce(
    ImagesProvider &images, std::shared_ptr<const ImageTree<int64_t>> root,
    const jxl::ModularOptions &options, const EncodingOptions &encoding_options,
    ProgressReporter *progress);

}  // namespace research

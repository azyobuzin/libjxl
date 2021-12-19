#pragma once

#include "cost_graph.h"
#include "enc.h"

namespace research {

// MSTから総当たりで、圧縮率が良くなるケースだけひとつの画像にまとめる
std::vector<EncodedImages> EncodeWithBruteForce(
    ImagesProvider &images, std::shared_ptr<const ImageTree<size_t>> root,
    const jxl::ModularOptions &options, size_t max_refs,
    ProgressReporter *progress);

}  // namespace research

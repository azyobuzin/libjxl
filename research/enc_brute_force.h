#pragma once

#include "cost_graph.h"
#include "enc.h"

namespace research {

std::vector<EncodedImages> EncodeWithBruteForce(
    ImagesProvider &images, std::shared_ptr<ImageTree<size_t>> root,
    const jxl::ModularOptions &options, ProgressReporter *progress);

}

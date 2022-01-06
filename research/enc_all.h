#pragma once

#include <filesystem>
#include <mlpack/core.hpp>
#include <string>

#include "images_provider.h"

namespace research {

arma::Row<size_t> ClusterImages(size_t split, float fraction,
                                const std::string& method, size_t k, int margin,
                                ImagesProvider& images);

void WriteIndexFile(uint32_t width, uint32_t height, uint32_t n_channel,
                    uint32_t n_clusters, const arma::Row<size_t>& assignments,
                    const std::filesystem::path& out_dir);

}  // namespace research

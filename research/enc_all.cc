#include "enc_all.h"

#include <tbb/parallel_for.h>

#include <mlpack/methods/kmeans/kmeans.hpp>

#include "cocbo.h"
#include "common_cluster.h"
#include "prop_extract.h"

namespace fs = std::filesystem;
using namespace mlpack::kmeans;

namespace research {

arma::Row<size_t> ClusterImages(size_t split, float fraction,
                                const std::string& method, size_t k, int margin,
                                ImagesProvider& images) {
  jxl::ModularOptions options{.nb_repeats = fraction};
  std::vector<uint32_t> props_to_use(std::cbegin(kPropsToUse),
                                     std::cend(kPropsToUse));
  jxl::TreeSamples tree_samples;

  // 量子化方法を決定するために適当なサンプリング
  SamplesForQuantization samples_for_quantization =
      CollectSamplesForQuantization(images, options);
  InitializeTreeSamples(tree_samples, props_to_use, options.max_property_values,
                        samples_for_quantization);

  // 特徴量計算
  arma::mat prop_mat;
  CreatePropertyMatrix(images, split, options, tree_samples, prop_mat);

  // クラスタリング
  arma::Row<size_t> assignments;
  if (method == "kmeans") {
    KMeans<> model;
    model.Cluster(prop_mat, k, assignments);
  } else if (method == "cocbo") {
    ClusterWithCocbo(prop_mat, k, std::max(static_cast<int>(k) - margin, 0),
                     k + 1 + margin, assignments);
  } else {
    JXL_ABORT("method is invalid");
  }

  JXL_ASSERT(assignments.size() == images.size());

  return assignments;
}

void WriteIndexFile(uint32_t width, uint32_t height, uint32_t n_channel,
                    uint32_t n_clusters, const arma::Row<size_t>& assignments,
                    const fs::path& out_dir) {
  IndexFields fields;
  fields.width = width;
  fields.height = height;
  fields.n_channel = n_channel;
  fields.n_clusters = n_clusters;
  fields.assignments.resize(assignments.size());
  std::copy(assignments.cbegin(), assignments.cend(),
            fields.assignments.begin());

  jxl::BitWriter writer;
  JXL_CHECK(jxl::Bundle::Write(fields, &writer, 0, nullptr));
  writer.ZeroPadToByte();
  jxl::Span<const uint8_t> span = writer.GetSpan();

  fs::path out_path = out_dir / "index.bin";
  FILE* fp = fopen(out_path.c_str(), "wb");
  bool failed =
      fwrite(span.data(), sizeof(uint8_t), span.size(), fp) != span.size();
  failed |= fclose(fp) != 0;

  if (failed) JXL_ABORT("Failed to write %s", out_path.c_str());
}

}  // namespace research

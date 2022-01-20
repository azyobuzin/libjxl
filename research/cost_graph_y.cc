#include <tbb/parallel_for.h>

#include <mlpack/core.hpp>
#include <opencv2/imgproc.hpp>

#include "cost_graph.h"
#include "enc_cluster.h"
#include "enc_flif.h"

using namespace jxl;

namespace research {

namespace {

typedef BidirectionalCostGraph<double> G;

inline size_t n_edges(size_t n_images) { return n_images * (n_images - 1); }

inline size_t n_jobs(size_t n_images) {
  return n_images + n_edges(n_images) / 2;
}

// 画像の1番目のチャネルの値をベクトルに代入する
template <typename Vec>
void ImageToVec(const Image &image, Vec &vec) {
  JXL_ASSERT(vec.is_vec());
  auto &c = image.channel[image.nb_meta_channels];
  vec.set_size(image.w * image.h);
  for (size_t r = 0; r < image.h; r++) {
    std::copy(c.Row(r), c.Row(r) + image.w, vec.begin() + (image.w * r));
  }
}

inline size_t DestinationIndexDiv2(size_t i, size_t n_images) {
  // 0 + n-1 + n-2 + ... + n-i = i*n - (0 + 1 + 2 + ... + i)
  return i * n_images - i * (i + 1) / 2;
}

}  // namespace

BidirectionalCostGraphResult<double> CreateGraphWithYDistance(
    ImagesProvider &ip, SelfCostMethod self_cost_method,
    const jxl::ModularOptions &options_in, ProgressReporter *progress) {
  JXL_CHECK(self_cost_method == kSelfCostJxl ||
            self_cost_method == kSelfCostFlif);

  const size_t n_images = ip.size();
  JXL_CHECK(n_images > 0);

  std::atomic_size_t completed_jobs = 0;
  std::vector<arma::vec> images(n_images);
  std::vector<double> self_costs(n_images);

  tbb::parallel_for(size_t(0), n_images, [&](size_t i) {
    auto image = std::make_shared<Image>(ip.get(i));

    // arma::vecに詰め替え
    ImageToVec(*image, images[i]);

    // self-cost 計算
    switch (self_cost_method) {
      case kSelfCostJxl: {
        auto ci = CombineImage(std::move(image));
        BitWriter writer;
        ModularOptions options = options_in;
        Tree tree = LearnTree(writer, ci, options, kParentReferenceNone);
        EncodeImages(writer, ci, options, kParentReferenceNone, tree);
        self_costs[i] = writer.BitsWritten();
      } break;

      case kSelfCostFlif: {
        cv::Mat mat = ip.GetBgr(i);
        cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
        self_costs[i] = ComputeEncodedBytesWithFlif(mat);
      } break;
    }

    size_t current_cj = ++completed_jobs;
    if (progress) progress->report(current_cj, n_jobs(n_images));
  });

  // グラフを作成する
  std::vector<std::pair<G::vertex_descriptor, G::vertex_descriptor>> edges(
      n_edges(n_images));
  std::vector<double> costs(n_edges(n_images));

  tbb::parallel_for(size_t(0), n_images - 1, [&](size_t i) {
    size_t dst_idx = DestinationIndexDiv2(i, n_images);

    if (i == 0) JXL_ASSERT(dst_idx == 0);

    for (size_t j = i + 1; j < n_images; j++) {
      // TODO(research): 無向グラフにしたい
      edges[dst_idx * 2] = {i, j};
      edges[dst_idx * 2 + 1] = {j, i};
      costs[dst_idx * 2] = costs[dst_idx * 2 + 1] =
          mlpack::metric::EuclideanDistance::Evaluate(images[i], images[j]);
      dst_idx++;

      size_t current_cj = ++completed_jobs;
      if (progress) progress->report(current_cj, n_jobs(n_images));
    }

    JXL_ASSERT(dst_idx == DestinationIndexDiv2(i + 1, n_images));
    if (i == n_images - 2) JXL_ASSERT(dst_idx == n_edges(n_images) / 2);
  });

  JXL_ASSERT(completed_jobs == n_jobs(n_images));

  return {std::move(self_costs),
          {edges.begin(), edges.end(), costs.begin(), n_images}};
}

}  // namespace research

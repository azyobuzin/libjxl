#include <tbb/parallel_for.h>

#include <mlpack/core.hpp>
#include <opencv2/imgproc.hpp>

#include "cost_graph.h"
#include "enc_cluster.h"
#include "enc_flif.h"

using namespace jxl;

namespace research {

typedef BidirectionalCostGraph<double> G;

BidirectionalCostGraphResult<double> CreateGraphWithRandomCost(
    ImagesProvider &ip, SelfCostMethod self_cost_method,
    const jxl::ModularOptions &options_in, ProgressReporter *progress) {
  JXL_CHECK(self_cost_method == kSelfCostJxl ||
            self_cost_method == kSelfCostFlif);

  const size_t n_images = ip.size();
  JXL_CHECK(n_images > 0);

  std::atomic_size_t completed_jobs = 0;
  std::vector<double> self_costs(n_images);

  tbb::parallel_for(size_t(0), n_images, [&](size_t i) {
    // self-cost 計算
    switch (self_cost_method) {
      case kSelfCostJxl: {
        auto ci = CombineImage(std::make_shared<Image>(ip.get(i)));
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
    if (progress) progress->report(current_cj, n_images);
  });

  // グラフを作成する
  const size_t n_edges = n_images * (n_images - 1);
  std::vector<std::pair<G::vertex_descriptor, G::vertex_descriptor>> edges;
  edges.reserve(n_edges);

  for (size_t i = 0; i < n_images; i++) {
    for (size_t j = 0; j < n_images; j++) {
      if (i != j) edges.emplace_back(i, j);
    }
  }

  arma::vec costs(n_edges, arma::fill::randu);

  return {std::move(self_costs),
          {edges.begin(), edges.end(), costs.begin(), n_images}};
}

}  // namespace research

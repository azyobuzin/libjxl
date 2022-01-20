#include <tbb/parallel_for.h>

#include <mlpack/core.hpp>
#include <opencv2/imgproc.hpp>

#include "cost_graph.h"
#include "enc_cluster.h"
#include "flif/library/flif.h"

using namespace jxl;

namespace research {

namespace {

typedef BidirectionalCostGraph<double> G;

inline size_t n_edges(size_t n_images) { return n_images * (n_images - 1); }

inline size_t n_jobs(size_t n_images) {
  return n_images + n_edges(n_images) / 2;
}

template <typename T>
void ImageToMat(const Image &image, arma::Mat<T> &mat) {
  auto &c = image.channel[image.nb_meta_channels];
  mat.set_size(image.h, image.w);
  for (size_t r = 0; r < image.h; r++) {
    std::copy(c.Row(r), c.Row(r) + image.w, mat.row(r).begin());
  }
}

G CreateGraphWithYRmse(const std::vector<arma::mat> &images,
                       ProgressReporter *progress,
                       std::atomic_size_t &completed_jobs) {
  const size_t n_images = images.size();
  std::vector<std::pair<G::vertex_descriptor, G::vertex_descriptor>> edges(
      n_edges(n_images));
  std::vector<double> costs(n_edges(n_images));

  tbb::parallel_for(size_t(0), n_images - 1, [&](size_t i) {
    size_t dst_idx = i * n_images - i * (i + 1);

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

    i++;
    JXL_ASSERT(dst_idx == i * n_images - i * (i + 1));

    if (i == n_images - 1) JXL_ASSERT(dst_idx == n_edges(n_images) / 2);
  });

  JXL_ASSERT(completed_jobs == n_jobs(n_images));

  return {edges.begin(), edges.end(), costs.begin(), n_images};
}

void CleanupFlifEncoder(FLIF_ENCODER **encoder) {
  flif_destroy_encoder(*encoder);
}

}  // namespace

BidirectionalCostGraphResult<double> CreateGraphWithYRmseAndJxlSelfCost(
    ImagesProvider &ip, const jxl::ModularOptions &options_in,
    ProgressReporter *progress) {
  const size_t n_images = ip.size();
  std::atomic_size_t completed_jobs = 0;

  std::vector<arma::mat> images(n_images);
  std::vector<double> self_costs(n_images);

  tbb::parallel_for(size_t(0), n_images, [&](size_t i) {
    auto image = std::make_shared<Image>(ip.get(i));

    // arma::matに詰め替え
    ImageToMat(*image, images[i]);

    // self-cost 計算
    auto ci = CombineImage(std::move(image));
    BitWriter writer;
    ModularOptions options = options_in;
    Tree tree = LearnTree(writer, ci, options, /*max_refs=*/0);
    EncodeImages(writer, ci, options, /*max_refs=*/0, tree);
    self_costs[i] = writer.BitsWritten();

    size_t current_cj = ++completed_jobs;
    if (progress) progress->report(current_cj, n_jobs(n_images));
  });

  return {std::move(self_costs),
          CreateGraphWithYRmse(images, progress, completed_jobs)};
}

BidirectionalCostGraphResult<double> CreateGraphWithYRmseAndFlifSelfCost(
    ImagesProvider &ip, ProgressReporter *progress) {
  const size_t n_images = ip.size();
  std::atomic_size_t completed_jobs = 0;

  std::vector<arma::mat> images(n_images);
  std::vector<double> self_costs(n_images);

  tbb::parallel_for(size_t(0), n_images, [&](size_t i) {
    // arma::matに詰め替え
    ImageToMat(ip.get(i), images[i]);

    // FLIFでエンコードしてバイト数を数える
    FLIF_ENCODER *__attribute__((cleanup(CleanupFlifEncoder))) encoder =
        flif_create_encoder();
    flif_encoder_set_crc_check(encoder, 0);
    cv::Mat mat = ip.GetBgr(i);
    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
    FLIF_IMAGE *fi =
        flif_import_image_RGB(mat.cols, mat.rows, mat.ptr(), mat.step);
    flif_encoder_add_image_move(encoder, fi);
    void *blob = nullptr;
    size_t blob_size = 0;
    JXL_CHECK(flif_encoder_encode_memory(encoder, &blob, &blob_size));
    self_costs[i] = blob_size;
    flif_free_memory(blob);

    size_t current_cj = ++completed_jobs;
    if (progress) progress->report(current_cj, n_jobs(n_images));
  });

  return {std::move(self_costs),
          CreateGraphWithYRmse(images, progress, completed_jobs)};
}

}  // namespace research

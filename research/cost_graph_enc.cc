#include <tbb/parallel_for.h>

#include "cost_graph.h"
#include "edmonds_optimum_branching.hpp"
#include "enc_jxl_multi.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"
#include "lib/jxl/modular/encoding/ma_common.h"

using namespace jxl;

namespace research {

namespace {

typedef BidirectionalCostGraph<size_t> G;

struct LearnedTree {
  Tree tree;
  size_t n_bits;
};

LearnedTree LearnTree(Image image, const ModularOptions &options) {
  BitWriter writer;
  Tree tree = LearnTree(writer, CombineImage(std::move(image)), options, 0);
  return {tree, writer.BitsWritten()};
}

size_t ComputeEncodedBits(Image image, const ModularOptions &options,
                          const Tree &tree) {
  BitWriter writer;
  EncodeImages(writer, CombineImage(std::move(image)), options, 0, tree);
  return writer.BitsWritten();
}

}  // namespace

BidirectionalCostGraphResult<size_t> CreateGraphWithDifferentTree(
    ImagesProvider &ip, const jxl::ModularOptions &options,
    ProgressReporter *progress) {
  const size_t n_images = ip.size();
  const size_t n_edges = n_images * (n_images - 1);
  const size_t n_jobs = n_edges + n_images;
  std::atomic_size_t completed_jobs = 0;

  // クラスタ単位はメモリにすべて乗り切る前提で
  std::vector<Image> images(n_images);

  // すべての決定木学習を行う
  std::vector<LearnedTree> learned_trees(n_images);
  tbb::parallel_for(size_t(0), n_images, [&](size_t i) {
    images[i] = ip.get(i);
    learned_trees[i] = LearnTree(images[i].clone(), options);
    completed_jobs++;
    if (progress) progress->report(completed_jobs, n_jobs);
  });

  // グラフを作成する
  std::vector<size_t> self_costs(n_images);
  std::vector<std::pair<G::vertex_descriptor, G::vertex_descriptor>> edges(
      n_edges);
  std::vector<size_t> costs(n_edges);

  tbb::parallel_for(size_t(0), n_images, [&](size_t i) {
    size_t dst_idx = (n_images - 1) * i;
    const auto &tree_lhs = learned_trees[i];

    // 自分自身の決定木で圧縮した場合
    self_costs[i] =
        tree_lhs.n_bits +
        ComputeEncodedBits(images[i].clone(), options, tree_lhs.tree);

    // この決定木で他の画像を圧縮した場合
    for (size_t j = 0; j < n_images; j++) {
      if (i == j) continue;

      edges[dst_idx] = {i, j};
      costs[dst_idx] =
          ComputeEncodedBits(images[j].clone(), options, tree_lhs.tree);
      dst_idx++;

      completed_jobs++;
      if (progress) progress->report(completed_jobs, n_jobs);
    }

    JXL_ASSERT(dst_idx == (n_images - 1) * (i + 1));
  });

  JXL_ASSERT(completed_jobs == n_jobs);

  return {std::move(self_costs),
          G(edges.begin(), edges.end(), costs.begin(), n_images)};
}

std::shared_ptr<ImageTree<size_t>> CreateMstWithDifferentTree(
    ImagesProvider &images, const jxl::ModularOptions &options,
    ProgressReporter *progress) {
  const size_t n_images = images.size();
  auto gr = CreateGraphWithDifferentTree(images, options, progress);
  auto &g = gr.graph;
  JXL_ASSERT(gr.self_costs.size() == n_images);
  JXL_ASSERT(num_vertices(g) == n_images);

  // 1枚ずつ圧縮したときにもっとも小さくなる画像を根とする
  std::array<G::vertex_descriptor, 1> roots = {static_cast<size_t>(
      std::min_element(gr.self_costs.cbegin(), gr.self_costs.cend()) -
      gr.self_costs.cbegin())};

  // 有向MST
  std::vector<G::edge_descriptor> edges;
  edges.reserve(n_images);
  edmonds_optimum_branching<false, true, true>(
      g, get(boost::vertex_index_t(), g), get(boost::edge_weight_t(), g),
      roots.begin(), roots.end(), std::back_inserter(edges));

  std::vector<std::shared_ptr<ImageTree<size_t>>> tree_nodes;
  tree_nodes.reserve(n_images);
  for (size_t i = 0; i < n_images; i++)
    tree_nodes.emplace_back(
        new ImageTree<size_t>{.image_idx = i, .self_cost = gr.self_costs[i]});

  for (auto &e : edges) {
    auto &src = tree_nodes.at(source(e, g));
    auto &tgt = tree_nodes.at(target(e, g));
    src->children.push_back(tgt);
    src->costs.push_back(get(boost::edge_weight_t(), g, e));
    JXL_ASSERT(!tgt->parent);  // parent must be null
    tgt->parent = src;
  }

  return tree_nodes.at(roots[0]);
}

}  // namespace research

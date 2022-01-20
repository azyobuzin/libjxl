#include <tbb/parallel_for.h>

#include "cost_graph.h"
#include "edmonds_optimum_branching.hpp"
#include "enc_cluster.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"
#include "lib/jxl/modular/encoding/ma_common.h"

using namespace jxl;

namespace research {

namespace {

typedef BidirectionalCostGraph<int64_t> G;

struct LearnedTree {
  Tree tree;
  int wp_mode;
  size_t n_bits;
};

LearnedTree LearnTree(std::shared_ptr<const Image> image,
                      const ModularOptions &options_in) {
  BitWriter writer;
  ModularOptions options = options_in;
  Tree tree = LearnTree(writer, CombineImage(std::move(image)), options, 0);
  return {tree, options.wp_mode, writer.BitsWritten()};
}

size_t ComputeEncodedBits(std::shared_ptr<const Image> image,
                          const ModularOptions &options, const Tree &tree) {
  BitWriter writer;
  EncodeImages(writer, CombineImage(std::move(image)), options, 0, tree);
  return writer.BitsWritten();
}

template <typename Cost>
ImageTree<Cost> ComputeMstFromGraphImpl(
    const BidirectionalCostGraphResult<Cost> &gr) {
  size_t n_images = gr.self_costs.size();
  const auto &g = gr.graph;

  JXL_CHECK(n_images <=
            static_cast<size_t>(std::numeric_limits<int32_t>::max()));
  JXL_ASSERT(num_vertices(g) == n_images);

  using VD =
      typename std::remove_reference<decltype(g)>::type::vertex_descriptor;
  using ED = typename std::remove_reference<decltype(g)>::type::edge_descriptor;
  static_assert(std::is_same<VD, size_t>::value,
                "vertex_descriptor must be size_t");

  // 1枚ずつ圧縮したときにもっとも小さくなる画像を根とする
  size_t root = std::min_element(gr.self_costs.cbegin(), gr.self_costs.cend()) -
                gr.self_costs.cbegin();
  std::array<VD, 1> roots = {root};

  // 有向MST
  std::vector<ED> edges;
  edges.reserve(n_images);
  edmonds_optimum_branching<false, true, true>(
      g, get(boost::vertex_index_t(), g), get(boost::edge_weight_t(), g),
      roots.begin(), roots.end(), std::back_inserter(edges));

  std::vector<ImageTreeNode<Cost>> tree_nodes;
  tree_nodes.reserve(n_images);
  for (size_t i = 0; i < n_images; i++) {
    tree_nodes.push_back(ImageTreeNode<Cost>{
        .image_idx = i, .self_cost = gr.self_costs[i], .parent = -1});
  }

  for (auto &e : edges) {
    auto src = source(e, g);
    auto tgt = target(e, g);
    tree_nodes.at(src).children.push_back(ImageTreeEdge<Cost>{
        static_cast<int32_t>(tgt), get(boost::edge_weight_t(), g, e)});
    auto &tgt_node = tree_nodes.at(tgt);
    JXL_ASSERT(tgt_node.parent == -1);
    tgt_node.parent = static_cast<int32_t>(src);
  }

  JXL_ASSERT(tree_nodes.at(root).parent == -1);
  return {std::move(tree_nodes), static_cast<int32_t>(root)};
}

}  // namespace

BidirectionalCostGraphResult<int64_t> CreateGraphWithDifferentTree(
    ImagesProvider &ip, const jxl::ModularOptions &options,
    ProgressReporter *progress) {
  const size_t n_images = ip.size();
  const size_t n_edges = n_images * (n_images - 1);
  const size_t n_jobs = n_edges + n_images;
  std::atomic_size_t completed_jobs = 0;

  // クラスタ単位はメモリにすべて乗り切る前提で
  std::vector<std::shared_ptr<const Image>> images(n_images);

  // すべての決定木学習を行う
  std::vector<LearnedTree> learned_trees(n_images);
  tbb::parallel_for(size_t(0), n_images, [&](size_t i) {
    images[i] = std::make_shared<Image>(ip.get(i));
    learned_trees[i] = LearnTree(images[i], options);
    completed_jobs++;
    if (progress) progress->report(completed_jobs, n_jobs);
  });

  // グラフを作成する
  std::vector<int64_t> self_costs(n_images);
  std::vector<std::pair<G::vertex_descriptor, G::vertex_descriptor>> edges(
      n_edges);
  std::vector<int64_t> costs(n_edges);

  tbb::parallel_for(size_t(0), n_images, [&](size_t i) {
    size_t dst_idx = (n_images - 1) * i;
    const auto &tree_self = learned_trees[i];
    ModularOptions local_options = options;
    local_options.wp_mode = tree_self.wp_mode;

    // 自分自身の決定木で圧縮した場合
    self_costs[i] =
        tree_self.n_bits +
        ComputeEncodedBits(images[i], local_options, tree_self.tree);

    // この画像を他の決定木で圧縮した場合
    for (size_t j = 0; j < n_images; j++) {
      if (i == j) continue;

      const auto &tree_other = learned_trees[j];
      local_options.wp_mode = tree_other.wp_mode;
      int64_t cost_bits =
          tree_other.n_bits +
          ComputeEncodedBits(images[i], local_options, tree_other.tree);
      edges[dst_idx] = {j, i};
      costs[dst_idx] = cost_bits - self_costs[i];
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

ImageTree<int64_t> ComputeMstFromGraph(
    const BidirectionalCostGraphResult<int64_t> &gr) {
  return ComputeMstFromGraphImpl<int64_t>(gr);
}

}  // namespace research

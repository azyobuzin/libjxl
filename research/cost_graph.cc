#include "cost_graph.h"

#include "edmonds_optimum_branching.hpp"

using namespace jxl;

namespace research {

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

ImageTree<int64_t> ComputeMstFromGraph(
    const BidirectionalCostGraphResult<int64_t> &gr) {
  return ComputeMstFromGraphImpl<int64_t>(gr);
}

ImageTree<double> ComputeMstFromGraph(
    const BidirectionalCostGraphResult<double> &gr) {
  return ComputeMstFromGraphImpl<double>(gr);
}

}  // namespace research

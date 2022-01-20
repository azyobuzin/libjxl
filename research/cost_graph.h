#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/adjacency_matrix.hpp>

#include "images_provider.h"
#include "lib/jxl/modular/options.h"
#include "progress.h"

namespace research {

// ある画像を利用して別の画像を圧縮するときのコストを表すグラフ
template <typename Cost>
using BidirectionalCostGraph =
    boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                          boost::no_property,
                          boost::property<boost::edge_weight_t, Cost>>;

template <typename Cost>
struct BidirectionalCostGraphResult {
  // 1枚だけで圧縮したときのコスト
  std::vector<Cost> self_costs;
  BidirectionalCostGraph<Cost> graph;
};

template <typename Cost>
struct ImageTreeEdge {
  int32_t target;
  Cost cost;
};

template <typename Cost>
struct ImageTreeNode {
  size_t image_idx;
  Cost self_cost;
  int32_t parent;
  std::vector<ImageTreeEdge<Cost>> children;
};

template <typename Cost>
struct ImageTree {
  std::vector<ImageTreeNode<Cost>> nodes;
  int32_t root;
};

// ある画像から学習した決定木を使って別の画像を圧縮したときのサイズを利用して、コストグラフを作成する。
BidirectionalCostGraphResult<int64_t> CreateGraphWithDifferentTree(
    ImagesProvider &images, const jxl::ModularOptions &options,
    ProgressReporter *progress);

// 1枚だけで圧縮したときのコストがもっとも小さい画像を根としてMSTを求める。
// edmonds_optimum_branching.hpp の扱いが厄介なので、 Cost をテンプレートにしないでオーバーロードにする。
ImageTree<int64_t> ComputeMstFromGraph(
    const BidirectionalCostGraphResult<int64_t> &gr);

}  // namespace research

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

// TODO(research): shared_ptrによる木ではなくvectorで実現したい
template <typename Cost>
struct ImageTree {
  size_t image_idx;
  Cost self_cost;
  std::shared_ptr<ImageTree> parent;
  std::vector<std::shared_ptr<ImageTree>> children;
  std::vector<Cost> costs;
};

// ある画像から学習した決定木を使って別の画像を圧縮したときのサイズを利用して、コストグラフを作成する。
BidirectionalCostGraphResult<int64_t> CreateGraphWithDifferentTree(
    ImagesProvider &images, const jxl::ModularOptions &options,
    ProgressReporter *progress);

// 1枚だけで圧縮したときのコストがもっとも小さい画像を根としてMSTを求める。
// edmonds_optimum_branching.hpp の扱いが厄介なので、 Cost をテンプレートにしないでオーバーロードにする。
std::shared_ptr<ImageTree<int64_t>> ComputeMstFromGraph(
    const BidirectionalCostGraphResult<int64_t> &gr);

}  // namespace research

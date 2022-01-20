#pragma once

#include <stack>

#include "cost_graph.h"
#include "enc_cluster.h"
#include "tbb/parallel_for_each.h"

namespace research {

namespace detail {

using namespace jxl;

struct EncodingTreeNode {
  EncodedCombinedImage encoded_image;
  int32_t parent;
  std::vector<int32_t> children;
};

EncodedCombinedImage ComputeEncodedBits(
    std::vector<std::shared_ptr<const Image>> &&images,
    std::vector<uint32_t> &&image_indices, std::vector<uint32_t> &&references,
    const ModularOptions &options_in, const EncodingOptions &encoding_options);

// MSTをとりあえず1枚ずつ圧縮した形式にする
template <typename Cost>
std::vector<EncodingTreeNode> CreateEncodingTree(
    const ImageTree<Cost> &tree, ImagesProvider &images,
    const ModularOptions &options, const EncodingOptions &encoding_options,
    ProgressReporter *progress) {
  JXL_CHECK(tree.nodes.size() == images.size());

  std::atomic_size_t n_completed = 0;
  std::vector<EncodedCombinedImage> encoded_data(images.size());

  JXL_CHECK(encoded_data.size() <=
            static_cast<size_t>(std::numeric_limits<int32_t>::max()));

  // 圧縮結果を用意する
  // 後ですべて使うので並列にやっておく
  tbb::parallel_for(
      uint32_t(0), static_cast<uint32_t>(encoded_data.size()), [&](uint32_t i) {
        encoded_data[i] =
            ComputeEncodedBits({std::make_shared<const Image>(images.get(i))},
                               {i}, {}, options, encoding_options);
        if (progress) progress->report(++n_completed, encoded_data.size() * 2);
      });

  std::vector<EncodingTreeNode> result_tree;
  result_tree.reserve(encoded_data.size());
  result_tree.emplace_back(EncodingTreeNode{
      std::move(encoded_data.at(tree.nodes.at(tree.root).image_idx)), -1});

  std::stack<std::pair<int32_t, int32_t>> stack;
  stack.emplace(tree.root, 0);

  while (!stack.empty()) {
    auto [src_node_idx, dst_node_idx] = stack.top();
    stack.pop();

    auto &src_node = tree.nodes.at(src_node_idx);
    auto &dst_node = result_tree.at(dst_node_idx);

    JXL_ASSERT(dst_node.children.empty());
    dst_node.children.reserve(src_node.children.size());

    // コストの小さい順を得る
    std::vector<std::pair<Cost, size_t>> costs;
    costs.reserve(src_node.children.size());
    for (size_t i = 0; i < src_node.children.size(); i++)
      costs.emplace_back(src_node.children[i].cost, i);
    std::sort(costs.begin(), costs.end());

    for (const auto &[cost, i] : costs) {
      int32_t child_idx = src_node.children[i].target;
      int32_t new_node_idx = result_tree.size();
      dst_node.children.push_back(new_node_idx);
      result_tree.emplace_back(EncodingTreeNode{
          std::move(encoded_data.at(tree.nodes.at(child_idx).image_idx)),
          dst_node_idx});
      stack.emplace(child_idx, new_node_idx);
    }
  }

  JXL_CHECK(result_tree.size() == encoded_data.size());

  return result_tree;
}

std::vector<EncodedCombinedImage> EncodeWithBruteForceCore(
    std::vector<EncodingTreeNode> &tree, const ModularOptions &options,
    const EncodingOptions &encoding_options, bool brute_force,
    ProgressReporter *progress);

}  // namespace detail

// MSTから総当たりで、圧縮率が良くなるケースだけひとつの画像にまとめる
template <typename Cost>
std::vector<EncodedCombinedImage> EncodeWithBruteForce(
    ImagesProvider &images, const ImageTree<Cost> &tree,
    const jxl::ModularOptions &options, const EncodingOptions &encoding_options,
    ProgressReporter *progress) {
  auto encoding_tree = detail::CreateEncodingTree<Cost>(
      tree, images, options, encoding_options, progress);
  return detail::EncodeWithBruteForceCore(encoding_tree, options,
                                          encoding_options, true, progress);
}

// MSTの行きがけ順で、画像をまとめて圧縮する
template <typename Cost>
std::vector<EncodedCombinedImage> EncodeWithCombineAll(
    ImagesProvider &images, const ImageTree<Cost> &tree,
    const jxl::ModularOptions &options, const EncodingOptions &encoding_options,
    ProgressReporter *progress) {
  auto encoding_tree = detail::CreateEncodingTree<Cost>(
      tree, images, options, encoding_options, progress);
  return detail::EncodeWithBruteForceCore(encoding_tree, options,
                                          encoding_options, false, progress);
}

}  // namespace research

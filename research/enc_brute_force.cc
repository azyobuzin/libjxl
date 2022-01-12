#include "enc_brute_force.h"

#include <stack>

#include "enc_cluster.h"
#include "enc_flif.h"
#include "lib/jxl/modular/transform/transform.h"
#include "tbb/concurrent_vector.h"
#include "tbb/parallel_for_each.h"

using namespace jxl;

namespace research {

namespace {

struct EncodingTreeNode {
  EncodedCombinedImage encoded_image;
  int32_t parent;
  std::vector<uint32_t> children;
};

EncodedCombinedImage ComputeEncodedBits(
    std::vector<std::shared_ptr<const Image>> &&images,
    std::vector<uint32_t> &&image_indices, std::vector<uint32_t> &&references,
    const ModularOptions &options_in, const EncodingOptions &encoding_options) {
  std::vector<std::shared_ptr<const Image>> jxl_images;

  if (encoding_options.flif_enabled) {
    jxl_images.reserve(images.size());
    for (const auto &image : images) {
      JXL_CHECK(image->nb_meta_channels == 0 && image->channel.size() == 3);
      Image y_image(image->w, image->h, image->bitdepth, 1);
      CopyImageTo(image->channel[0].plane, &y_image.channel[0].plane);
      jxl_images.emplace_back(new Image(std::move(y_image)));
    }
  } else {
    jxl_images = images;
  }

  CombinedImage ci = CombineImage(jxl_images, references);
  BitWriter writer;
  ModularOptions options = options_in;
  Tree tree = LearnTree(writer, ci, options, encoding_options.parent_reference);
  EncodeImages(writer, ci, options, encoding_options.parent_reference, tree);
  writer.ZeroPadToByte();

  PaddedBytes flif_data;
  if (encoding_options.flif_enabled) {
    flif_data =
        EncodeColorSignalWithFlif(images, encoding_options.flif_learn_repeats,
                                  encoding_options.flif_additional_props);
  }

  return {std::move(image_indices), std::move(images), std::move(references),
          std::move(writer).TakeBytes(), std::move(flif_data)};
}

// MSTをとりあえず1枚ずつ圧縮した形式にする
std::vector<EncodingTreeNode> CreateEncodingTree(
    std::shared_ptr<const ImageTree<size_t>> root, ImagesProvider &images,
    const ModularOptions &options, const EncodingOptions &encoding_options,
    ProgressReporter *progress) {
  std::atomic_size_t n_completed = 0;
  std::vector<EncodedCombinedImage> encoded_data(images.size());

  JXL_CHECK(encoded_data.size() <=
            static_cast<size_t>(std::numeric_limits<int32_t>().max()));

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
  result_tree.emplace_back(
      EncodingTreeNode{std::move(encoded_data.at(root->image_idx)), -1});

  std::stack<std::pair<std::shared_ptr<const ImageTree<size_t>>, uint32_t>>
      stack;
  stack.emplace(root, 0);

  while (!stack.empty()) {
    auto [src_node, dst_node_idx] = stack.top();
    stack.pop();

    JXL_CHECK(src_node->children.size() == src_node->costs.size());

    auto &dst_node = result_tree.at(dst_node_idx);
    JXL_ASSERT(dst_node.children.empty());
    dst_node.children.reserve(src_node->children.size());

    // コストの小さい順を得る
    std::vector<std::pair<size_t, size_t>> costs;
    costs.reserve(src_node->costs.size());
    for (size_t i = 0; i < src_node->costs.size(); i++)
      costs.emplace_back(src_node->costs[i], i);
    std::sort(costs.begin(), costs.end());

    for (const auto &[cost, i] : costs) {
      const auto &child = src_node->children[i];
      uint32_t new_node_idx = result_tree.size();
      dst_node.children.push_back(new_node_idx);
      result_tree.emplace_back(
          EncodingTreeNode{std::move(encoded_data.at(child->image_idx)),
                           static_cast<int32_t>(dst_node_idx)});
      stack.emplace(child, new_node_idx);
    }
  }

  JXL_CHECK(result_tree.size() == encoded_data.size());

  return result_tree;
}

struct Traverse {
  std::vector<EncodingTreeNode> &tree;
  const ModularOptions &options;
  const EncodingOptions &encoding_options;
  ProgressReporter *progress;
  tbb::concurrent_vector<EncodedCombinedImage> results;
  std::atomic_size_t n_completed;

  Traverse(std::vector<EncodingTreeNode> &tree, const ModularOptions &options,
           const EncodingOptions &encoding_options, ProgressReporter *progress)
      : tree(tree),
        options(options),
        encoding_options(encoding_options),
        progress(progress),
        results(),
        n_completed(0) {
    results.reserve(tree.size());
  }

  void operator()(uint32_t node_idx) {
    const size_t n_images = tree.size();
    auto &node = tree.at(node_idx);

    // 子孫要素をすべて処理
    tbb::parallel_for_each(node.children.begin(), node.children.end(),
                           [this](uint32_t child) { (*this)(child); });

    for (uint32_t child_idx : node.children) {
      // node に child を結合した場合に、圧縮率が改善するか試す
      // TODO(research): 最後は並列化が効かなくなって、すごく遅い
      auto &child = tree.at(child_idx);

      auto images = node.encoded_image.included_images;
      images.insert(images.end(), child.encoded_image.included_images.cbegin(),
                    child.encoded_image.included_images.cend());

      auto image_indices = node.encoded_image.image_indices;
      image_indices.insert(image_indices.end(),
                           child.encoded_image.image_indices.cbegin(),
                           child.encoded_image.image_indices.cend());

      auto references = node.encoded_image.references;
      uint32_t ref_base = node.encoded_image.references.size();
      references.reserve(references.size() + ref_base + 1);
      references.push_back(0);  // 子の最初の画像は親を参照する
      // 子の参照を修正する
      for (uint32_t r : child.encoded_image.references)
        references.push_back(ref_base + r);

      EncodedCombinedImage combined_bits =
          ComputeEncodedBits(std::move(images), std::move(image_indices),
                             std::move(references), options, encoding_options);

      if (combined_bits.n_bytes() <
          node.encoded_image.n_bytes() + child.encoded_image.n_bytes()) {
        node.encoded_image = std::move(combined_bits);

        // メモリ解放
        child.encoded_image = {};
      } else {
        // 効果がないので、単独で出力
        n_completed += child.encoded_image.image_indices.size();
        results.push_back(std::move(child.encoded_image));
        if (progress) progress->report(n_completed + n_images, n_images * 2);
      }
    }

    // 根ならばこれ以上戻れないので出力
    if (node.parent < 0) {
      n_completed += node.encoded_image.image_indices.size();
      results.push_back(std::move(node.encoded_image));
      if (progress) progress->report(n_completed + n_images, n_images * 2);
    }
  }
};

}  // namespace

std::vector<EncodedCombinedImage> EncodeWithBruteForce(
    ImagesProvider &images, std::shared_ptr<const ImageTree<size_t>> root,
    const ModularOptions &options, const EncodingOptions &encoding_options,
    ProgressReporter *progress) {
  auto tree =
      CreateEncodingTree(root, images, options, encoding_options, progress);
  Traverse traverse(tree, options, encoding_options, progress);
  traverse(0);

  std::vector<EncodedCombinedImage> results;
  results.reserve(traverse.results.size());
  std::move(traverse.results.begin(), traverse.results.end(),
            std::back_inserter(results));

  // インデックスがあまりランダムな順番にならないといいなぁ（願望）
  std::sort(results.begin(), results.end(),
            [](const EncodedCombinedImage &x, const EncodedCombinedImage &y) {
              return x.image_indices.at(0) < y.image_indices.at(0);
            });

  return results;
}

}  // namespace research

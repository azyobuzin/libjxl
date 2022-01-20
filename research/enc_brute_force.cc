#include "enc_brute_force.h"

#include <stack>

#include "enc_cluster.h"
#include "enc_flif.h"
#include "lib/jxl/modular/transform/transform.h"
#include "tbb/concurrent_vector.h"
#include "tbb/parallel_for_each.h"

using namespace jxl;

namespace research::detail {

EncodedCombinedImage ComputeEncodedBits(
    std::vector<std::shared_ptr<const Image>> &&images,
    std::vector<uint32_t> &&image_indices, std::vector<uint32_t> &&references,
    const ModularOptions &options_in, const EncodingOptions &encoding_options) {
  std::vector<std::shared_ptr<const Image>> jxl_images;

  if (encoding_options.flif_enabled) {
    jxl_images.reserve(images.size());
    for (const auto &image : images) {
      JXL_CHECK(image->nb_meta_channels == 0 && image->channel.size() == 3);
      auto y_image =
          std::make_shared<Image>(image->w, image->h, image->bitdepth, 1);
      CopyImageTo(image->channel[0].plane, &y_image->channel[0].plane);
      jxl_images.push_back(std::move(y_image));
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

namespace {

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
      auto &child = tree.at(child_idx);

      // 探索後の子は捨てられているはず
      JXL_ASSERT(child.children.empty());

      // node に child を結合した場合に、圧縮率が改善するか試す
      // TODO(research): 最後は並列化が効かなくなって、すごく遅い
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

    node.children.clear();

    // 根ならばこれ以上戻れないので出力
    if (node.parent < 0) {
      n_completed += node.encoded_image.image_indices.size();
      results.push_back(std::move(node.encoded_image));
      if (progress) progress->report(n_completed + n_images, n_images * 2);
    }
  }
};

EncodedCombinedImage EncodeWithCombineAllCore(
    std::vector<EncodingTreeNode> &tree, const ModularOptions &options,
    const EncodingOptions &encoding_options) {
  std::vector<std::shared_ptr<const Image>> images;
  images.reserve(tree.size());
  std::vector<uint32_t> image_indices;
  image_indices.reserve(tree.size());
  std::vector<uint32_t> references;
  references.reserve(tree.size() - 1);

  // ノードが何フレーム目に挿入されたかを記録する
  std::vector<uint32_t> frame_idx_by_node_idx(tree.size());

  // 行きがけ順を求める
  std::stack<int32_t> stack;
  stack.push(0);

  while (!stack.empty()) {
    int32_t node_idx = stack.top();
    stack.pop();

    frame_idx_by_node_idx.at(node_idx) = images.size();

    const auto &node = tree.at(node_idx);
    images.insert(images.end(), node.encoded_image.included_images.begin(),
                  node.encoded_image.included_images.end());
    image_indices.insert(image_indices.end(),
                         node.encoded_image.image_indices.begin(),
                         node.encoded_image.image_indices.end());

    if (node.parent >= 0) {
      references.push_back(frame_idx_by_node_idx.at(node.parent));
    }

    for (auto x : node.children) stack.push(x);
  }

  return ComputeEncodedBits(std::move(images), std::move(image_indices),
                            std::move(references), options, encoding_options);
}

}  // namespace

std::vector<EncodedCombinedImage> EncodeWithBruteForceCore(
    std::vector<EncodingTreeNode> &tree, const ModularOptions &options,
    const EncodingOptions &encoding_options, bool brute_force,
    ProgressReporter *progress) {
  std::vector<EncodedCombinedImage> results;

  if (brute_force) {
    Traverse traverse(tree, options, encoding_options, progress);
    traverse(0);

    results.reserve(traverse.results.size());
    std::move(traverse.results.begin(), traverse.results.end(),
              std::back_inserter(results));

    // インデックスがあまりランダムな順番にならないといいなぁ（願望）
    std::sort(results.begin(), results.end(),
              [](const EncodedCombinedImage &x, const EncodedCombinedImage &y) {
                return x.image_indices.at(0) < y.image_indices.at(0);
              });
  } else {
    results.reserve(1);
    results.push_back(
        EncodeWithCombineAllCore(tree, options, encoding_options));

    if (progress) {
      const size_t n_images = tree.size();
      const size_t n_jobs = n_images * 2;
      progress->report(n_jobs, n_jobs);
    }
  }

  return results;
}

}  // namespace research::detail

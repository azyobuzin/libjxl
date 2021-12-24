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

struct EncodingTree {
  EncodedCombinedImage images;
  std::shared_ptr<EncodingTree> parent;
  std::vector<std::shared_ptr<EncodingTree>> children;
};

EncodedCombinedImage ComputeEncodedBits(
    std::vector<std::shared_ptr<const jxl::Image>> images,
    std::vector<uint32_t> image_indices, const jxl::ModularOptions &options,
    const EncodingOptions &encoding_options) {
  std::vector<std::shared_ptr<const jxl::Image>> jxl_images;

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

  CombinedImage ci = CombineImage(jxl_images);
  BitWriter writer;
  Tree tree = LearnTree(writer, ci, options, encoding_options.max_refs);
  EncodeImages(writer, ci, options, encoding_options.max_refs, tree);
  size_t n_bits = writer.BitsWritten();
  writer.ZeroPadToByte();

  PaddedBytes flif_data;
  if (encoding_options.flif_enabled) {
    flif_data =
        EncodeColorSignalWithFlif(images, encoding_options.flif_learn_repeats,
                                  encoding_options.flif_additional_props);
  }

  return {std::move(image_indices), std::move(images),
          std::move(writer).TakeBytes(), n_bits, std::move(flif_data)};
}

// MSTをとりあえず1枚ずつ圧縮した形式にする
std::shared_ptr<EncodingTree> CreateEncodingTree(
    std::shared_ptr<const ImageTree<size_t>> root, ImagesProvider &images,
    const jxl::ModularOptions &options, const EncodingOptions &encoding_options,
    ProgressReporter *progress) {
  std::atomic_size_t n_completed = 0;
  std::vector<EncodedCombinedImage> encoded_data(images.size());

  JXL_CHECK(encoded_data.size() <= std::numeric_limits<uint32_t>().max());

  // 圧縮結果を用意する
  // 後ですべて使うので並列にやっておく
  tbb::parallel_for(
      uint32_t(0), static_cast<uint32_t>(encoded_data.size()), [&](uint32_t i) {
        encoded_data[i] = ComputeEncodedBits(
            {std::make_shared<const jxl::Image>(images.get(i))}, {i}, options,
            encoding_options);
        if (progress) progress->report(++n_completed, encoded_data.size() * 2);
      });

  std::shared_ptr<EncodingTree> result_root(
      new EncodingTree{std::move(encoded_data.at(root->image_idx))});

  std::stack<std::pair<std::shared_ptr<const ImageTree<size_t>>,
                       std::shared_ptr<EncodingTree>>>
      stack;
  stack.push({root, result_root});

  while (!stack.empty()) {
    auto [src_node, dst_node] = stack.top();
    stack.pop();

    JXL_CHECK(src_node->children.size() == src_node->costs.size());
    dst_node->children.reserve(src_node->children.size());

    // コストの小さい順を得る
    std::vector<std::pair<size_t, size_t>> costs;
    costs.reserve(src_node->costs.size());
    for (size_t i = 0; i < src_node->costs.size(); i++)
      costs.emplace_back(src_node->costs[i], i);
    std::sort(costs.begin(), costs.end());

    for (const auto &[cost, i] : costs) {
      const auto &child = src_node->children[i];
      auto &new_node = dst_node->children.emplace_back(new EncodingTree{
          std::move(encoded_data.at(child->image_idx)), dst_node});
      stack.push({child, new_node});
    }
  }

  return result_root;
}

struct Traverse {
  size_t n_images;
  const jxl::ModularOptions &options;
  const EncodingOptions &encoding_options;
  ProgressReporter *progress;
  tbb::concurrent_vector<EncodedCombinedImage> results;
  std::atomic_size_t n_completed;

  Traverse(size_t n_images, const jxl::ModularOptions &options,
           const EncodingOptions &encoding_options, ProgressReporter *progress)
      : n_images(n_images),
        options(options),
        encoding_options(encoding_options),
        progress(progress),
        results(),
        n_completed(0) {
    results.reserve(n_images);
  }

  void operator()(std::shared_ptr<EncodingTree> node) {
    // 子孫要素をすべて処理
    tbb::parallel_for_each(
        node->children.begin(), node->children.end(),
        [this](std::shared_ptr<EncodingTree> &child) { (*this)(child); });

    for (auto &child : node->children) {
      // 探索後の子は捨てられているはず
      JXL_ASSERT(child->children.empty());

      // node に child を結合した場合に、圧縮率が改善するか試す
      // TODO(research): 最後は並列化が効かなくなって、すごく遅い
      auto images = node->images.included_images;
      images.insert(images.end(), child->images.included_images.cbegin(),
                    child->images.included_images.cend());
      auto image_indices = node->images.image_indices;
      image_indices.insert(image_indices.end(),
                           child->images.image_indices.cbegin(),
                           child->images.image_indices.cend());
      EncodedCombinedImage combined_bits =
          ComputeEncodedBits(std::move(images), std::move(image_indices),
                             options, encoding_options);

      if (combined_bits.n_bytes() <
          node->images.n_bytes() + child->images.n_bytes()) {
        node->images = std::move(combined_bits);
      } else {
        // 効果がないので、単独で出力
        n_completed += child->images.image_indices.size();
        results.push_back(std::move(child->images));
        progress->report(n_completed + n_images, n_images * 2);
      }
    }

    node->children.clear();

    // 根ならばこれ以上戻れないので出力
    if (!node->parent) {
      n_completed += node->images.image_indices.size();
      results.push_back(std::move(node->images));
      progress->report(n_completed + n_images, n_images * 2);
    }
  }
};

}  // namespace

std::vector<EncodedCombinedImage> EncodeWithBruteForce(
    ImagesProvider &images, std::shared_ptr<const ImageTree<size_t>> root,
    const jxl::ModularOptions &options, const EncodingOptions &encoding_options,
    ProgressReporter *progress) {
  Traverse traverse(images.size(), options, encoding_options, progress);
  traverse(
      CreateEncodingTree(root, images, options, encoding_options, progress));

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

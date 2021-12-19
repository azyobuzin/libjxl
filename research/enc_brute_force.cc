#include "enc_brute_force.h"

#include <stack>

#include "enc_jxl_multi.h"
#include "lib/jxl/modular/transform/transform.h"
#include "tbb/concurrent_vector.h"
#include "tbb/parallel_for_each.h"

using namespace jxl;

namespace research {

namespace {

struct EncodingTree {
  EncodedImages images;
  std::shared_ptr<EncodingTree> parent;
  std::vector<std::shared_ptr<EncodingTree>> children;
};

EncodedImages ComputeEncodedBits(
    std::vector<std::shared_ptr<const jxl::Image>> images,
    std::vector<size_t> image_indices, const jxl::ModularOptions &options,
    size_t max_refs) {
  CombinedImage ci = CombineImage(images);
  BitWriter writer;
  Tree tree = LearnTree(writer, ci, options, max_refs);
  EncodeImages(writer, ci, options, max_refs, tree);
  size_t n_bits = writer.BitsWritten();
  writer.ZeroPadToByte();
  return {std::move(image_indices), std::move(images),
          std::move(writer).TakeBytes(), n_bits};
}

// TODO: CreateEncodingTree に時間がかかるので、進捗表示の改善が必要。あと並列化

// MSTをとりあえず1枚ずつ圧縮した形式にする
std::shared_ptr<EncodingTree> CreateEncodingTree(
    std::shared_ptr<const ImageTree<size_t>> root, ImagesProvider &images,
    const jxl::ModularOptions &options) {
  std::shared_ptr<EncodingTree> result_root(new EncodingTree{
      ComputeEncodedBits(
          {std::make_shared<const jxl::Image>(images.get(root->image_idx))},
          {root->image_idx}, options, 0),
  });

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
          ComputeEncodedBits({std::make_shared<const jxl::Image>(
                                 images.get(child->image_idx))},
                             {child->image_idx}, options, 0),
          dst_node});
      stack.push({child, new_node});
    }
  }

  return result_root;
}

struct Traverse {
  size_t n_images;
  const jxl::ModularOptions &options;
  size_t max_refs;
  ProgressReporter *progress;
  tbb::concurrent_vector<EncodedImages> results;
  std::atomic_size_t n_completed;

  Traverse(size_t n_images, const jxl::ModularOptions &options, size_t max_refs,
           ProgressReporter *progress)
      : n_images(n_images),
        options(options),
        max_refs(max_refs),
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
      // TODO: 最後は並列化が効かなくなって、すごく遅い
      auto images = node->images.included_images;
      images.insert(images.end(), child->images.included_images.cbegin(),
                    child->images.included_images.cend());
      auto image_indices = node->images.image_indices;
      image_indices.insert(image_indices.end(),
                           child->images.image_indices.cbegin(),
                           child->images.image_indices.cend());
      EncodedImages combined_bits = ComputeEncodedBits(
          std::move(images), std::move(image_indices), options, max_refs);

      if (combined_bits.n_bits < node->images.n_bits + child->images.n_bits) {
        node->images = std::move(combined_bits);
      } else {
        // 効果がないので、単独で出力
        results.push_back(std::move(child->images));
      }
    }

    node->children.clear();

    // 根ならばこれ以上戻れないので出力
    if (!node->parent) results.push_back(std::move(node->images));

    if (progress) progress->report(++n_completed, n_images);
  }
};

}  // namespace

std::vector<EncodedImages> EncodeWithBruteForce(
    ImagesProvider &images, std::shared_ptr<const ImageTree<size_t>> root,
    const jxl::ModularOptions &options, size_t max_refs,
    ProgressReporter *progress) {
  Traverse traverse(images.size(), options, max_refs, progress);
  traverse(CreateEncodingTree(root, images, options));

  std::vector<EncodedImages> results;
  results.reserve(traverse.results.size());
  std::move(traverse.results.begin(), traverse.results.end(),
            std::back_inserter(results));
  return results;
}

}  // namespace research

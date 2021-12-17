#include "enc_brute_force.h"

namespace research {

namespace {

// TODO(research): 1枚を圧縮するやつ
// TODO(research): 複数枚を圧縮するやつ

struct EncodingTree {
  EncodedImages images;
  std::shared_ptr<EncodingTree> parent;
  std::vector<std::shared_ptr<EncodingTree>> children;

  static std::shared_ptr<EncodingTree> FromImageTree(
      std::shared_ptr<ImageTree<size_t>> root) {
    // std::shared_ptr<EncodingTree> result_root(new EncodingTree{images})
    // TODO(research)
  }
};

EncodedImages RemoveFromTree(std::shared_ptr<EncodingTree> &node) {
  JXL_ASSERT(node->children.empty());

  if (node->parent) {
    // 親から削除する
    node->parent->children.erase(std::find(node->parent->children.begin(),
                                           node->parent->children.end(), node));
  }

  JXL_ASSERT(node.use_count() == 1);
  return std::move(node->images);
}

}  // namespace

std::vector<EncodedImages> EncodeWithBruteForce(
    ImagesProvider &images, std::shared_ptr<ImageTree<size_t>> root,
    const jxl::ModularOptions &options, ProgressReporter *progress) {
  // TODO(research)
}

}  // namespace research

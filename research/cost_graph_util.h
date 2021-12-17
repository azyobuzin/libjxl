#include <ostream>
#include <stack>

#include "cost_graph.h"

// clang-format off
// edmonds_optimum_branching_impl.hpp の後に include する
#include <boost/graph/graphviz.hpp>
// clang-format on

namespace research {

// ImageTree を DOT (Graphviz) 形式で出力する
template <typename Cost>
void PrintImageTreeDot(std::ostream &dst, std::shared_ptr<ImageTree<Cost>> root,
                       ImagesProvider *images) {
  std::stack<std::shared_ptr<ImageTree<Cost>>> nodes;
  nodes.push(std::move(root));

  dst << "digraph G {" << std::endl;

  while (!nodes.empty()) {
    auto node = nodes.top();
    nodes.pop();

    if (images) {
      dst << node->image_idx << " [label="
          << boost::escape_dot_string(images->get_label(node->image_idx))
          << "];" << std::endl;
    }

    JXL_ASSERT(node->children.size() == node->costs.size());

    for (size_t i = 0; i < node->children.size(); i++) {
      auto &child = node->children[i];
      dst << node->image_idx << "->" << child->image_idx
          << " [label=" << boost::escape_dot_string(node->costs[i]) << "];"
          << std::endl;
      nodes.push(child);
    }
  }

  dst << "}" << std::endl;
}

}  // namespace research

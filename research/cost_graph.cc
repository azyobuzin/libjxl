#include "cost_graph.h"

#include <stack>

namespace research {

void PrintImageTreeDot(std::ostream& dst, std::shared_ptr<ImageTree> root,
                       ImagesProvider* images) {
  std::stack<std::shared_ptr<ImageTree>> nodes;
  nodes.push(std::move(root));

  dst << "digraph G {" << std::endl;

  while (!nodes.empty()) {
    auto node = nodes.top();
    nodes.pop();

    if (images) {
      dst << node->image_idx << " [label=\""
          << images->get_label(node->image_idx) << "\"];" << std::endl;
    }

    JXL_ASSERT(node->children.size() == node->costs.size());

    for (size_t i = 0; i < node->children.size(); i++) {
      auto& child = node->children[i];
      dst << node->image_idx << "->" << child->image_idx << " [label=\""
          << node->costs[i] << "\"];" << std::endl;
      nodes.push(child);
    }
  }

  dst << "}" << std::endl;
}

}  // namespace research

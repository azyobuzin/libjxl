#pragma once

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
void PrintImageTreeDot(std::ostream& dst, const ImageTree<Cost>& tree,
                       ImagesProvider* images) {
  std::stack<int32_t> stack;
  stack.push(tree.root);

  dst << "digraph G {" << std::endl;

  while (!stack.empty()) {
    const auto& node = tree.nodes.at(static_cast<size_t>(stack.top()));
    stack.pop();

    if (images) {
      dst << node.image_idx << " [label="
          << boost::escape_dot_string(images->get_label(node.image_idx)) << "];"
          << std::endl;
    }

    for (const auto& edge : node.children) {
      dst << node.image_idx << "->" << tree.nodes.at(edge.target).image_idx
          << " [label=" << boost::escape_dot_string(edge.cost) << "];"
          << std::endl;
      stack.push(edge.target);
    }
  }

  dst << "}" << std::endl;
}

}  // namespace research

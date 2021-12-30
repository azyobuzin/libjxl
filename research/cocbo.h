#pragma once

#include <mlpack/core.hpp>

namespace research {

// COCBO法によるクラスタリング
// https://www.jstage.jst.go.jp/article/fss/32/0/32_329/_article/-char/ja/
// https://ieeexplore.ieee.org/document/8023341
void ClusterWithCocbo(const arma::mat& data, size_t k, size_t lower_bound,
                      size_t upper_bound, arma::Row<size_t>& assignments,
                      size_t max_iterations = 1000);

}  // namespace research

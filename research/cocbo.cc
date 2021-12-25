#include "cocbo.h"

#include <fmt/core.h>
#include <glpk.h>

#include <mlpack/methods/kmeans/sample_initialization.hpp>

#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/status.h"

namespace research {

void ClusterWithCocbo(const arma::mat& data, size_t k, size_t lower_bound,
                      size_t upper_bound, arma::Row<size_t>& assignments,
                      size_t max_iterations) {
  JXL_CHECK(lower_bound > 0 && lower_bound <= k);
  JXL_CHECK(upper_bound >= k + 1);

  size_t n_cluster = std::max<size_t>(data.n_cols / k, 1);
  JXL_CHECK(n_cluster <= data.n_cols);

  glp_prob* lp = glp_create_prob();
  glp_set_prob_name(lp, "COCBO");
  glp_set_obj_dir(lp, GLP_MIN);  // 最小化問題

  // 帰属度 u_ki
  glp_add_cols(lp, data.n_cols * n_cluster);
  for (size_t k = 0; k < data.n_cols; k++) {
    for (size_t i = 0; i < n_cluster; i++) {
      std::string col_name = fmt::format("u_{},{}", k, i);
      glp_set_col_name(lp, k * n_cluster + i + 1, col_name.c_str());
      glp_set_col_kind(lp, k * n_cluster + i + 1, GLP_BV);  // 0 or 1
    }
  }

  glp_add_rows(lp, data.n_cols * n_cluster);

  {
    std::vector<int> indices(data.n_cols + 1);
    std::vector<double> ones(data.n_cols + 1, 1);

    // 制約1: 各要素（k）はいずれかのクラスタに属している
    for (size_t k = 0; k < data.n_cols; k++) {
      int row_idx = k + 1;
      std::string row_name = fmt::format("sum(u_{},i)=1", k);
      glp_set_row_name(lp, row_idx, row_name.c_str());
      glp_set_row_bnds(lp, row_idx, GLP_FX, 1, 1);
      for (size_t i = 0; i < n_cluster; i++)
        indices[i + 1] = k * n_cluster + i + 1;
      glp_set_mat_row(lp, row_idx, n_cluster, indices.data(), ones.data());
    }

    // 制約2: 各クラスタには [lower_bound, upper_bound] 個の要素が属している
    for (size_t i = 0; i < n_cluster; i++) {
      int row_idx = data.n_cols + i + 1;
      std::string row_name =
          fmt::format("{} <= sum(u_k,{}) <= {}", lower_bound, i, upper_bound);
      glp_set_row_name(lp, row_idx, row_name.c_str());
      glp_set_row_bnds(lp, row_idx, GLP_DB, lower_bound, upper_bound);
      for (size_t k = 0; k < data.n_cols; k++)
        indices[k + 1] = k * n_cluster + i + 1;
      glp_set_mat_row(lp, row_idx, data.n_cols, indices.data(), ones.data());
    }
  }

  arma::mat centroids;
  mlpack::kmeans::SampleInitialization().Cluster(data, n_cluster, centroids);
  arma::mat new_centroids(centroids.n_rows, centroids.n_cols, arma::fill::none);
  std::vector<size_t> assign_count(n_cluster);
  mlpack::metric::EuclideanDistance metric;

  assignments.zeros(data.n_cols);

  for (size_t i = 0; i < max_iterations; i++) {
    // 目的関数を設定
    for (size_t k = 0; k < data.n_cols; k++) {
      for (size_t i = 0; i < n_cluster; i++) {
        double distance = metric.Evaluate(data.col(k), centroids.col(i));
        glp_set_obj_coef(lp, k * n_cluster + i + 1, distance);
      }
    }

    // 最適化
    int solve_result = glp_simplex(lp, nullptr);
    if (solve_result != 0) JXL_ABORT("glp_simplex returned %d", solve_result);

    // 結果取得
    for (size_t k = 0; k < data.n_cols; k++) {
      for (size_t i = 0;; i++) {
        auto u = glp_get_col_prim(lp, k * n_cluster + i + 1);
        JXL_DASSERT(u == 0 || u == 1);
        if (u == 1) {
          assignments[k] = i;
          break;
        }
        if (i >= n_cluster - 1) {
          JXL_ABORT("データ %" PRIuS " はいずれのクラスタにも属していません",
                    k);
        }
      }
    }

    // クラスタ中心の更新
    new_centroids.zeros();
    std::fill(assign_count.begin(), assign_count.end(), 0);
    for (size_t k = 0; k < data.n_cols; k++) {
      size_t i = assignments[k];
      assign_count[i]++;
      new_centroids.col(i) += data.col(k);
    }
    for (size_t i = 0; i < n_cluster; i++) {
      new_centroids.col(i) /= assign_count[i];
    }
    if (arma::approx_equal(new_centroids, centroids, "absdiff", 1e-5)) {
      // クラスタ中心が更新されなければ終わり
      break;
    }
    std::swap(centroids, new_centroids);
  }

  glp_delete_prob(lp);
}

}  // namespace research

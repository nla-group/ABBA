/*
Modified version of the C++ code from https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html
 */

#include <string>
#include <iostream>
#include <cassert>
#include <cmath>

#include "Ckmeans.1d.dp.h"


void fill_dp_matrix(const std::vector<double> & x, // data
                    std::vector< std::vector< double > > & S,
                    std::vector< std::vector< size_t > > & J,
                    const std::string & method)
  /*
   x: One dimension vector to be clustered, must be sorted (in any order).
   S: K x N matrix. S[q][i] is the sum of squares of the distance from
   each x[i] to its cluster mean when there are exactly x[i] is the
   last point in cluster q
   J: K x N backtrack matrix

   NOTE: All vector indices in this program start at position 0
   */
{
  const int K = (int) S.size();
  const int N = (int) S[0].size();

  std::vector<double> sum_x(N), sum_x_sq(N);

  std::vector<int> jseq;

  double shift = x[N/2]; // median. used to shift the values of x to
  //  improve numerical stability

  sum_x[0] = x[0] - shift;
  sum_x_sq[0] = (x[0] - shift) * (x[0] - shift);

  S[0][0] = 0;
  J[0][0] = 0;

  for(int i = 1; i < N; ++i) {

    sum_x[i] = sum_x[i-1] + x[i] - shift;
    sum_x_sq[i] = sum_x_sq[i-1] + (x[i] - shift) * (x[i] - shift);

    // Initialize for q = 0
    S[0][i] = dissimilarity(0, i, sum_x, sum_x_sq);
    J[0][i] = 0;
  }

  for(int q = 1; q < K; ++q) {
    int imin;
    if(q < K - 1) {
      imin = std::max(1, q);
    } else {
      // No need to compute S[K-1][0] ... S[K-1][N-2]
      imin = N-1;
    }

    if(method == "linear") {
      fill_row_q_SMAWK(imin, N-1, q, S, J, sum_x, sum_x_sq);
    } else if(method == "loglinear") {
      fill_row_q_log_linear(imin, N-1, q, q, N-1, S, J, sum_x, sum_x_sq);
    } else if(method == "quadratic") {
      fill_row_q(imin, N-1, q, S, J, sum_x, sum_x_sq);
    } else {
      throw std::string("ERROR: unknown method") + method + "!";
    }
  }
}

void backtrack(const std::vector<double> & x,
               const std::vector< std::vector< size_t > > & J,
               int* cluster, double* centers, double* withinss,
               double* count /*int* count*/)
{
  const size_t K = J.size();
  const size_t N = J[0].size();
  size_t cluster_right = N-1;
  size_t cluster_left;

  // Backtrack the clusters from the dynamic programming matrix
  for(int q = ((int)K)-1; q >= 0; --q) {
    cluster_left = J[q][cluster_right];

    for(size_t i = cluster_left; i <= cluster_right; ++i)
      cluster[i] = q;

    double sum = 0.0;

    for(size_t i = cluster_left; i <= cluster_right; ++i)
      sum += x[i];

    centers[q] = sum / (cluster_right-cluster_left+1);

    for(size_t i = cluster_left; i <= cluster_right; ++i)
      withinss[q] += (x[i] - centers[q]) * (x[i] - centers[q]);

    count[q] = (int) (cluster_right - cluster_left + 1);

    if(q > 0) {
      cluster_right = cluster_left - 1;
    }
  }
}

void backtrack(const std::vector<double> & x,
               const std::vector< std::vector< size_t > > & J,
               std::vector<size_t> & count, const int K)
{
  // const int K = (int) J.size();
  const size_t N = J[0].size();
  size_t cluster_right = N-1;
  size_t cluster_left;

  // Backtrack the clusters from the dynamic programming matrix
  for(int q = K-1; q >= 0; --q) {
    cluster_left = J[q][cluster_right];
    count[q] = cluster_right - cluster_left + 1;
    if(q > 0) {
      cluster_right = cluster_left - 1;
    }
  }
}

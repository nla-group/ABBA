/*
Modified version of the C++ code from https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html
 */

#include "Ckmeans.1d.dp.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <cstring>

template <class ForwardIterator>
size_t numberOfUnique(ForwardIterator first, ForwardIterator last)
{
  size_t nUnique;

  if (first == last) {
    nUnique = 0;
  } else {
    nUnique = 1;
    for (ForwardIterator itr=first+1; itr!=last; ++itr) {
      if (*itr != *(itr -1)) {
        nUnique ++;
      }
    }
  }
  return nUnique;
}

static const double * px;
bool compi(size_t i, size_t j)
{
  return px[i] < px[j];
}


Output kmeans_1d_dp(std::vector<double> x_v, size_t Kmin, size_t Kmax,
                                  double var, const std::string & method)
{
  const size_t N = x_v.size();
  const double* x = &x_v[0];

  int* cluster = new int[N];
  double* centres = new double[Kmax];
  double* withinss = new double[Kmax];
  double* size = new double[Kmax];
  double* BIC = new double[Kmax-Kmin+1];


  // Input:
  //  x -- an array of double precision numbers, not necessarily sorted
  //  Kmin -- the minimum number of clusters expected
  //  Kmax -- the maximum number of clusters expected
  // NOTE: All vectors in this program is considered starting at position 0.


  // Sort x
  std::vector<size_t> order(N);
  for(size_t i=0; i<order.size(); ++i) {
    order[i] = i;
  }
  bool is_sorted(true);
  for(size_t i=0; i<N-1; ++i) {
    if(x[i] > x[i+1]) {
      is_sorted = false;
      break;
    }
  }
  std::vector<double> x_sorted(x, x+N);
  if(! is_sorted) {
    px = x;
    std::sort(order.begin(), order.end(), compi);

    for(size_t i=0ul; i<order.size(); ++i) {
      x_sorted[i] = x[order[i]];
    }
  }

  // Find number of unique values
  const size_t nUnique = numberOfUnique(x_sorted.begin(), x_sorted.end());

  // Adjust Kmax according to nUnique
  Kmax = nUnique < Kmax ? nUnique : Kmax;
  Kmin = std::min(nUnique, Kmin);
  size_t Kopt;

  if(nUnique > 1) { // The case when not all elements are equal.

    std::vector< std::vector< double > > S( Kmax, std::vector<double>(N) );
    std::vector< std::vector< size_t > > J( Kmax, std::vector<size_t>(N) );

    fill_dp_matrix(x_sorted, S, J, method);

    // Fill in dynamic programming matrix
    // Choose an optimal number of levels between Kmin and Kmax
    Kopt = select_levels(x_sorted, J, Kmin, Kmax, BIC, var);

    if (Kopt < Kmax) { // Reform the dynamic programming matrix S and J
      J.erase(J.begin() + Kopt, J.end());
    }

    std::vector<int> cluster_sorted(N);

    // Backtrack to find the clusters beginning and ending indices
    backtrack(x_sorted, J, &cluster_sorted[0], centres, withinss, size);

    for(size_t i = 0; i < N; ++i) {
      // Obtain clustering on data in the original order
      cluster[order[i]] = cluster_sorted[i];
    }

  } else {  // A single cluster that contains all elements
    Kopt = 1;
    for(size_t i=0; i<N; ++i) {
      cluster[i] = 0;
    }
    centres[0] = x[0];
    withinss[0] = 0.0;
    size[0] = N;
  }

  std::vector<int> cluster_v(cluster, cluster+N);
  std::vector<double> centres_v(centres, centres+Kopt);
  std::vector<double> withinss_v(withinss, withinss+Kopt);
  std::vector<double> size_v(size, size+Kopt);
  std::vector<double> BIC_v(BIC, BIC+Kmax-Kmin+1);

  Output output = Output(cluster_v, centres_v, withinss_v, size_v, BIC_v, Kopt);

  delete[] cluster;
  delete[] centres;
  delete[] withinss;
  delete[] size;
  delete[] BIC;

  return output;

}  //end of kmeans_1d_dp()

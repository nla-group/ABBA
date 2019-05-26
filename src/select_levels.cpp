/*
Modified version of the C++ code from https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>

#include "Ckmeans.1d.dp.h"

#ifndef M_PI
const double M_PI = 3.14159265359;
#endif

void shifted_data_variance(const std::vector<double> & x,
                           const size_t left,
                           const size_t right,
                           double & mean, double & variance)
{
  double sum = 0.0;
  double sumsq = 0.0;

  mean = 0.0;
  variance = 0.0;

  size_t n = right - left + 1;

  if(right >= left) {

    double median = x[(left + right) / 2];

    for (size_t i = left; i <= right; ++i) {
      sum += x[i] - median;
      sumsq += (x[i] - median) * (x[i] - median);
    }
    mean = sum / n + median;

    if (n > 1) {
      /* changed from sample var, used to divide by n-1  */
      variance = (sumsq - sum * sum / n) / (n);
    }
  }
}

void range_of_variance(const std::vector<double> & x,
                       double & variance_min, double & variance_max)
{
  double dposmin = x[x.size()-1] - x[0];
  double dposmax = 0;

  for(size_t n=1; n<x.size(); ++n) {
    double d = x[n] - x[n-1];
    if(d > 0 && dposmin > d) {
      dposmin = d;
    }
    if(d > dposmax) {
      dposmax = d;
    }
  }
  variance_min = dposmin*dposmin/3.0;
  variance_max = dposmax*dposmax;
}


// Choose an optimal number of levels between Kmin and Kmax
size_t select_levels(const std::vector<double> & x,
                     const std::vector< std::vector< size_t > > & J,
                     size_t Kmin, size_t Kmax,
                     double * BIC, double var)
{
  const size_t N = x.size();

  if (Kmin > Kmax || N < 2) {
    return std::min(Kmin, Kmax);
  }

  size_t Kopt = Kmin;
  double maxBIC = (0.0);

  std::vector<double> lambda(Kmax);
  std::vector<double> mu(Kmax);
  std::vector<double> sigma2(Kmax);
  std::vector<double> variance(Kmax);
  std::vector<double> coeff(Kmax);

  for(size_t K = Kmin; K <= Kmax; ++K) {

    // Backtrack the matrix to determine boundaries between the bins.
    std::vector<size_t> size(K);
    backtrack(x, J, size, (int)K);

    size_t indexLeft = 0;
    size_t indexRight;

    for (size_t k = 0; k < K; ++k) { // Estimate GMM parameters first
      lambda[k] = size[k] / (double) N;

      indexRight = indexLeft + size[k] - 1;

      shifted_data_variance(x, indexLeft, indexRight, mu[k], sigma2[k]);
      variance[k] = sigma2[k];

      if(sigma2[k] == 0 || size[k] == 1) {

        double dmin;

        if(indexLeft > 0 && indexRight < N-1) {
          dmin = std::min(x[indexLeft] - x[indexLeft-1], x[indexRight+1] - x[indexRight]);
        } else if(indexLeft > 0) {
          dmin = x[indexLeft] - x[indexLeft-1];
        } else {
          dmin = x[indexRight+1] - x[indexRight];
        }
        if(sigma2[k] == 0) sigma2[k] = dmin * dmin / 4.0 / 9.0 ;
        if(size[k] == 1) sigma2[k] = dmin * dmin;
      }

      coeff[k] = lambda[k] / std::sqrt(2.0 * M_PI * sigma2[k]);

      indexLeft = indexRight + 1;
    }

    if(*std::max_element(std::begin(variance), std::end(variance)) < var){
      return K;
    }

    if (K == Kmin) {
      Kopt = Kmin;
    }
  }

  return Kmax;
 }

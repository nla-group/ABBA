/*
Modified version of the C++ code from https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html
 */

#include <vector>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <stack>
#include <cmath>

#include "Ckmeans.1d.dp.h"

void fill_row_q(int imin, int imax, int q,
                std::vector< std::vector<double> > & S,
                std::vector< std::vector<size_t> > & J,
                const std::vector<double> & sum_x,
                const std::vector<double> & sum_x_sq)
{
  // Assumption: each cluster must have at least one point.
  for(int i=imin; i<=imax; ++i) {
    S[q][i] = S[q-1][i-1];
    J[q][i] = i;
    int jmin = std::max(q, (int)J[q-1][i]);
    for(int j=i-1; j>=jmin; --j) {
      double Sj(S[q-1][j-1] +
        dissimilarity(j, i, sum_x, sum_x_sq)
      );

      if(Sj < S[q][i]) {
        S[q][i] = Sj;
        J[q][i] = j;
      }
    }
  }
}

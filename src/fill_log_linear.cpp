/*
Modified version of the C++ code from https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html
 */

#include <vector>
#include <iostream>
#include "Ckmeans.1d.dp.h"

void fill_row_q_log_linear(int imin, int imax, int q,
                           int jmin, int jmax,
                           std::vector< std::vector<double> > & S,
                           std::vector< std::vector<size_t> > & J,
                           const std::vector<double> & sum_x,
                           const std::vector<double> & sum_x_sq)
{
  if(imin > imax) {
    return;
  }

  const int N = (int) S[0].size();

  int i = (imin + imax) / 2;

  // Initialization of S[q][i]:
  S[q][i] = S[q - 1][i - 1];
  J[q][i] = i;

  int jlow=q; // the lower end for j

  if(imin > q) {
    // jlow = std::max(jlow, (int)J[q][imin-1]);
    jlow = std::max(jlow, jmin);
  }
  jlow = std::max(jlow, (int)J[q-1][i]);

  int jhigh = i - 1; // the upper end for j
  if(imax < N-1) {
    // jhigh = std::min(jhigh, (int)J[q][imax+1]);
    jhigh = std::min(jhigh, jmax);
  }

  for(int j=jhigh; j>=jlow; --j) {

    // compute s(j,i)
    double sji = dissimilarity(j, i, sum_x, sum_x_sq);

    // MS May 11, 2016 Added:
    if(sji + S[q-1][jlow-1] >= S[q][i]) break;

    // Examine the lower bound of the cluster border
    // compute s(jlow, i)
    double sjlowi =
      dissimilarity(jlow, i, sum_x, sum_x_sq);

    double SSQ_jlow = sjlowi + S[q-1][jlow-1];

    if(SSQ_jlow < S[q][i]) {
      // shrink the lower bound
      S[q][i] = SSQ_jlow;
      J[q][i] = jlow;
    }
    jlow ++;

    double SSQ_j = sji + S[q - 1][j - 1];
    if(SSQ_j < S[q][i]) {
      S[q][i] = SSQ_j;
      J[q][i] = j;
    }
  }

  jmin = (imin > q) ? (int)J[q][imin-1] : q;
  jmax = (int)J[q][i];

  fill_row_q_log_linear(imin, i-1, q, jmin, jmax,
                        S, J, sum_x, sum_x_sq);

  jmin = (int)J[q][i];
  jmax = (imax < N-1) ? (int)J[q][imax+1] : imax;
  fill_row_q_log_linear(i+1, imax, q, jmin, jmax,
                        S, J, sum_x, sum_x_sq);

}

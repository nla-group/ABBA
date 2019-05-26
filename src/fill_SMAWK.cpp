/*
Modified version of the C++ code from https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html
 */

#include <vector>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <stack>

#include "Ckmeans.1d.dp.h"

void reduce_in_place(int imin, int imax, int istep, int q,
                     const std::vector<size_t> & js,
                     std::vector<size_t> & js_red,
                     const std::vector< std::vector<double> > & S,
                     const std::vector< std::vector<size_t> > & J,
                     const std::vector<double> & sum_x,
                     const std::vector<double> & sum_x_sq)
{
  int N = (imax - imin) / istep + 1;

  js_red = js;

  if(N >= js.size()) {
    return;
  }

  // Two positions to move candidate j's back and forth
  int left = -1; // points to last favorable position / column
  int right = 0; // points to current position / column

  size_t m = js_red.size();

  while(m > N) { // js_reduced has more than N positions / columns

    int p = (left + 1);

    int i = (imin + p * istep);
    size_t j = (js_red[right]);

    double Sl = (S[q-1][j-1] +
      dissimilarity(j, i, sum_x, sum_x_sq));

    size_t jplus1 = (js_red[right+1]);
    double Slplus1 = (S[q-1][jplus1-1] +
      dissimilarity(jplus1, i, sum_x, sum_x_sq));

    if(Sl < Slplus1 && p < N-1) {
      js_red[ ++ left ] = j; // i += istep;
      right ++; // move on to next position / column p+1
    } else if(Sl < Slplus1 && p == N-1) {
      js_red[ ++ right ] = j; // delete position / column p+1
      m --;
    } else { // (Sl >= Slplus1)
      if(p > 0) { // i > imin
        // delete position / column p and
        //   move back to previous position / column p-1:
        js_red[right] = js_red[left --];
        // p --; // i -= istep;
      } else {
        right ++; // delete position / column 0
      }
      m --;
    }
  }

  for(int r=(left+1); r < m; ++r) {
    js_red[r] = js_red[right++];
  }

  js_red.resize(m);
  return;
}

inline void fill_even_positions
  (int imin, int imax, int istep, int q,
   const std::vector<size_t> & js,
   std::vector< std::vector<double> > & S,
   std::vector< std::vector<size_t> > & J,
   const std::vector<double> & sum_x,
   const std::vector<double> & sum_x_sq)
{
  // Derive j for even rows (0-based)
  size_t n = (js.size());
  int istepx2 = (istep << 1);
  size_t jl = (js[0]);
  for(int i=(imin), r(0); i<=imax; i+=istepx2) {

    // auto jmin = (i == imin) ? js[0] : J[q][i - istep];

    while(js[r] < jl) {
      // Increase r until it points to a value of at least jmin
      r ++;
    }

    // Initialize S[q][i] and J[q][i]
    S[q][i] = S[q-1][js[r]-1] +
      dissimilarity(js[r], i, sum_x, sum_x_sq);
    J[q][i] = js[r]; // rmin

    // Look for minimum S upto jmax within js
    int jh = (i + istep <= imax)
      ? J[q][i + istep] : js[n-1];

    int jmax = std::min((int)jh, (int)i);

    double sjimin(
        dissimilarity(jmax, i, sum_x, sum_x_sq)
      );

    for(++ r; r < n && js[r]<=jmax; r++) {

      const size_t & jabs = js[r];

      if(jabs > i) break;

      if(jabs < J[q-1][i]) continue;

      double s =
        dissimilarity(jabs, i, sum_x, sum_x_sq);
      double Sj = (S[q-1][jabs-1] + s);

      if(Sj <= S[q][i]) {
        S[q][i] = Sj;
        J[q][i] = js[r];
      } else if(S[q-1][jabs-1] + sjimin > S[q][i]) {
        break;
      }
    }
    r --;
    jl = jh;
  }
}

inline void find_min_from_candidates
  (int imin, int imax, int istep, int q,
   const std::vector<size_t> & js,
   std::vector< std::vector<double> > & S,
   std::vector< std::vector<size_t> > & J,
   const std::vector<double> & sum_x,
   const std::vector<double> & sum_x_sq)
{
  size_t rmin_prev = (0);

  for(int i=(imin); i<=imax; i+=istep) {

    size_t rmin = (rmin_prev);

    // Initialization of S[q][i] and J[q][i]
    S[q][i] = S[q-1][js[rmin] - 1] +
      dissimilarity(js[rmin], i, sum_x, sum_x_sq);
    J[q][i] = js[rmin];

    for(size_t r = (rmin+1); r<js.size(); ++r) {

      const size_t & j_abs = (js[r]);

      if(j_abs < J[q-1][i]) continue;
      if(j_abs > i) break;

      double Sj = (S[q-1][j_abs - 1] +
        dissimilarity(j_abs, i, sum_x, sum_x_sq));
      if(Sj <= S[q][i]) {
        S[q][i] = Sj;
        J[q][i] = js[r];
        rmin_prev = r;
      }
    }
  }
}

void SMAWK
  (int imin, int imax, int istep, int q,
   const std::vector<size_t> & js,
   std::vector< std::vector<double> > & S,
   std::vector< std::vector<size_t> > & J,
   const std::vector<double> & sum_x,
   const std::vector<double> & sum_x_sq)
{

  if(imax - imin <= 0 * istep) { // base case only one element left

    find_min_from_candidates(
      imin, imax, istep, q, js, S, J, sum_x, sum_x_sq);

  } else {

    std::vector<size_t> js_odd;

    reduce_in_place(imin, imax, istep, q, js, js_odd,
                    S, J, sum_x, sum_x_sq);

    int istepx2 = (istep << 1);
    int imin_odd = (imin + istep);
    int imax_odd = (imin_odd + (imax - imin_odd) / istepx2 * istepx2);

    // Recursion on odd rows (0-based):
    SMAWK(imin_odd, imax_odd, istepx2,
          q, js_odd, S, J, sum_x, sum_x_sq);

    fill_even_positions(imin, imax, istep, q, js,
                        S, J, sum_x, sum_x_sq);
  }
}

void fill_row_q_SMAWK(int imin, int imax, int q,
                      std::vector< std::vector<double> > & S,
                      std::vector< std::vector<size_t> > & J,
                      const std::vector<double> & sum_x,
                      const std::vector<double> & sum_x_sq)
{
  // Assumption: each cluster must have at least one point.

  std::vector<size_t> js(imax-q+1);
  int abs = (q);
  std::generate(js.begin(), js.end(), [&] { return abs++; } );

  SMAWK(imin, imax, 1, q, js, S, J, sum_x, sum_x_sq);
}

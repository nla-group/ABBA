/*
Modified version of the C++ code from https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html
 */

#include <cstddef> // For size_t
#include <vector>
#include <string>

class Output {
public:
  std::vector<int> cluster;
  std::vector<double> centres;
  std::vector<double> withinss;
  std::vector<double> size;
  std::vector<double> BIC;
  size_t Kopt;
  Output(std::vector<int> in_cluster, std::vector<double> in_centres, std::vector<double> in_withinss, std::vector<double> in_size, std::vector<double> in_BIC, size_t in_Kopt)
  {
    cluster = in_cluster;
    centres = in_centres;
    withinss = in_withinss;
    size = in_size;
    BIC = in_BIC;
    Kopt = in_Kopt;
  }
};

void fill_dp_matrix(
    const std::vector<double> & x,
    std::vector< std::vector< double > > & S,
    std::vector< std::vector< size_t > > & J,
    const std::string & method);

void backtrack(
    const std::vector<double> & x,
    const std::vector< std::vector< size_t > > & J,
    int* cluster, double* centres, double* withinss,
    double* count /*int* count*/);

void backtrack(
    const std::vector<double> & x,
    const std::vector< std::vector< size_t > > & J,
    std::vector<size_t> & count);

void fill_row_q_SMAWK(
    int imin, int imax, int q,
    std::vector< std::vector<double> > & S,
    std::vector< std::vector<size_t> > & J,
    const std::vector<double> & sum_x,
    const std::vector<double> & sum_x_sq);

void fill_row_q(
    int imin, int imax, int q,
    std::vector< std::vector<double> > & S,
    std::vector< std::vector<size_t> > & J,
    const std::vector<double> & sum_x,
    const std::vector<double> & sum_x_sq);


void fill_row_q_log_linear(
    int imin, int imax, int q, int jmin, int jmax,
    std::vector< std::vector<double> > & S,
    std::vector< std::vector<size_t> > & J,
    const std::vector<double> & sum_x,
    const std::vector<double> & sum_x_sq);

/* One-dimensional cluster algorithm implemented in C++ */
/* x is input one-dimensional vector and
 Kmin and Kmax stand for the range for the number of clusters*/
Output kmeans_1d_dp(std::vector<double> x_v,size_t Kmin, size_t Kmax,
                   double var, const std::string & method);

void backtrack(
    const std::vector<double> & x,
    const std::vector< std::vector< size_t > > & J,
    std::vector<size_t> & counts, const int K);

size_t select_levels(
    const std::vector<double> & x,
    const std::vector< std::vector< size_t > > & J,
    size_t Kmin, size_t Kmax, double *BIC, double var);

void range_of_variance(
    const std::vector<double> & x,
    double & variance_min, double & variance_max);

inline double dissimilarity(
    const size_t j, const size_t i,
    const std::vector<double> & sum_x, // running sum of xi
    const std::vector<double> & sum_x_sq // running sum of xi^2
)
{
  double sji(0.0);

  if(j >= i) {
    sji = 0.0;
  } else if(j > 0) {
    double muji = (sum_x[i] - sum_x[j-1]) / (i - j + 1);
    sji = sum_x_sq[i] - sum_x_sq[j-1] - (i - j + 1) * muji * muji;
  } else {
    sji = sum_x_sq[i] - sum_x[i] * sum_x[i] / (i+1);
  }

  sji = (sji < 0) ? 0 : sji;
  return sji;
}

/* Ckmeans.i */

%module Ckmeans

%include <std_vector.i>
%include <std_string.i>
%include <typemaps.i>

%{
#include "Ckmeans.1d.dp.h"
#include <vector>
#include <utility>
#include <string>
%}

%template(double_vector) std::vector<double>;
%template(int_vector) std::vector<int>;

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

Output kmeans_1d_dp(std::vector<double> x_v, size_t Kmin, size_t Kmax, double var, const std::string & method);

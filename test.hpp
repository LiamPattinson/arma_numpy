// test.hpp
// Simple tests to get the project going

#include <armadillo>

// 1D

double sum_vec( arma::vec v); // handle input of vec and output of something easily recognised
arma::sword sum_vec(arma::ivec v); // handle input of ivec and output of an armadillo typedef, plus handle overloading
arma::vec get_vec( int size ); // given something easily recognised, output vec
arma::vec reverse_vec( arma::vec v); // given a vec, return a vec
arma::icolvec reverse_vec( arma::ivec v); // given an ivec, return a typedef over ivec
arma::uvec reverse_vec( arma::uvec v); // given a uvec, return a uvec. Tests if numpy casting works properly.

// 2D

double sum_mat( arma::mat m);
arma::sword sum_mat( arma::imat m);
arma::mat transpose_mat( arma::mat m);

// 3D

double sum_cube( arma::cube c);
arma::sword sum_cube( arma::icube c);
arma::cube do_nothing( arma::cube c);
arma::mat get_second_slice( arma::cube c);

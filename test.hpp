// test.hpp
// Simple tests to get the project going

#include <armadillo>

double sum_vec( arma::vec v); // handle input of vec and output of something easily recognised
arma::sword sum_vec(arma::ivec v); // handle input of ivec and output of an armadillo typedef, plus handle overloading
arma::vec get_vec( int size ); // given something easily recognised, output vec
arma::vec reverse_vec( arma::vec v); // given a vec, return a vec
arma::icolvec reverse_vec( arma::ivec v); // given an ivec, return a typedef over ivec

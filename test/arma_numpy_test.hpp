// arma_numpy_test.hpp
// A collection of simple C++ functions to test arma_numpy.i

#include <armadillo>

// 1D

double sum_vec( arma::vec v); // handle input of vec and output of something easily recognised
arma::sword sum_vec(arma::ivec v); // handle input of ivec and output of an armadillo typedef, plus handle overloading
arma::vec get_vec( int size ); // given something easily recognised, output vec
arma::vec reverse_vec( arma::vec v); // given a vec, return a vec
arma::icolvec reverse_vec( arma::ivec v); // given an ivec, return a typedef over ivec
arma::uvec reverse_vec( arma::uvec v); // given a uvec, return a uvec. Tests if numpy casting works properly.

double sum_vec_by_ref( const arma::vec& v);
double sum_vec_by_ptr( const arma::vec* v);
void set_to_zero_by_ref( arma::vec& v);
void set_to_zero_by_ptr( arma::vec* v);

void print_memptr_by_val( arma::vec v);
void print_memptr_by_ref( arma::vec& v);
void print_memptr_by_ptr( arma::vec* v);
void print_memptr_by_const_ref( const arma::vec& v);
void print_memptr_by_const_ptr( const arma::vec* v);


// 2D

double sum_mat( arma::mat m);
arma::sword sum_mat( arma::imat m);
arma::mat transpose_mat( arma::mat m);
void set_to_zero_by_ref( arma::mat& m);
void set_to_zero_by_ptr( arma::mat* m);
double sum_mat_by_const_ref( const arma::mat& m);
double sum_mat_by_const_ptr( const arma::mat* m);

void print_memptr_by_val( arma::mat m);
void print_memptr_by_ref( arma::mat& m);
void print_memptr_by_ptr( arma::mat* m);
void print_memptr_by_const_ref( const arma::mat& m);
void print_memptr_by_const_ptr( const arma::mat* m);

// 3D

double sum_cube( arma::cube c);
arma::sword sum_cube( arma::icube c);
arma::cube do_nothing( arma::cube c);
arma::mat get_second_slice( arma::cube c);

void print_memptr_by_val( arma::cube c);
void print_memptr_by_ref( arma::cube& c);
void print_memptr_by_ptr( arma::cube* c);
void print_memptr_by_const_ref( const arma::cube& c);
void print_memptr_by_const_ptr( const arma::cube* c);

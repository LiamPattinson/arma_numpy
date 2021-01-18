#include "arma_numpy_test.hpp"
#include <cstdio>

double sum_vec( arma::vec v){
    return arma::sum(v);
}

arma::sword sum_vec( arma::ivec v){
    return arma::sum(v);
}

arma::vec get_vec( int size ){
    return arma::vec(size,arma::fill::zeros);
}

arma::vec reverse_vec( arma::vec v){
    return arma::reverse(v);
}

arma::ivec reverse_vec( arma::ivec v){
    return arma::reverse(v);
}

arma::uvec reverse_vec( arma::uvec v){
    return arma::reverse(v);
}

double sum_vec_by_ref( const arma::vec& v){
    return arma::sum(v);
}

double sum_vec_by_ptr( const arma::vec* v){
    return arma::sum(*v);
}

void set_to_zero_by_ref( arma::vec& v){
    v.zeros();
}

void set_to_zero_by_ptr( arma::vec* v){
    v->zeros();
}

std::size_t get_memptr( arma::vec* v){
    return (std::size_t)v->memptr();
}

std::size_t get_memptr_const( const arma::vec* v){
    return (std::size_t)v->memptr();
}

arma::vec& get_static_vec(){
    static arma::vec v = arma::vec(3,arma::fill::zeros);
    return v;
}

double sum_mat( arma::mat m){
    return arma::accu(m);    
}

arma::sword sum_mat( arma::imat m){
    return arma::accu(m);    
}

arma::mat transpose_mat( arma::mat m){
    return m.t();
}

void set_to_zero_by_ref( arma::mat& m){
    m.zeros();
}

void set_to_zero_by_ptr( arma::mat* m){
    m->zeros();
}

double sum_mat_by_const_ref( const arma::mat& m){
    return arma::accu(m);    
}

double sum_mat_by_const_ptr( const arma::mat* m){
    return arma::accu(*m);    
}

std::size_t get_memptr(arma::mat* m){
    return (std::size_t)m->memptr();
}

std::size_t get_memptr_const(const arma::mat* m){
    return (std::size_t)m->memptr();
}

double sum_cube( arma::cube c){
    return arma::accu(c);    
}

arma::sword sum_cube( arma::icube c){
    return arma::accu(c);
}

arma::cube do_nothing( arma::cube c){
    return c;
}

arma::mat get_second_slice( arma::cube c){
    return c.slice(1);
}

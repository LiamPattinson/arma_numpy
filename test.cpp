#include "test.hpp"

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

double sum_mat( arma::mat m){
    return arma::sum(arma::sum(m));    
}

arma::sword sum_mat( arma::imat m){
    return arma::sum(arma::sum(m));    
}

arma::mat transpose_mat( arma::mat m){
    return m.t();
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

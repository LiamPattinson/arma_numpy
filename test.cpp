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

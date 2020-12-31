#include "test.hpp"

double sum_vec( arma::vec v){
    return arma::sum(v);
}

arma::vec get_vec( int size ){
    return arma::vec(size,arma::fill::zeros);
}

arma::vec reverse_vec( arma::vec v){
    return arma::reverse(v);
}

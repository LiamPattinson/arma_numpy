// arma_numpy.i
// A SWIG interface file to convert between Armadillo vectors/matrices and Numpy arrays.

%header %{
    #include <algorithm>
    #include "numpy/arrayobject.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

// To start, a hardcoded method to convert between np.array and arma::vec

// typemap(in) converts from np.array to arma::vec
%typemap(in, fragment="NumPy_Fragments") arma::vec {
    int is_new;
    PyArrayObject* array = obj_to_array_allow_conversion( $input, NPY_DOUBLE, &is_new);
    $1 = arma::Col<double>( (double*) array_data(array), array_dimensions(array)[0], true);
}

// typemap(out) converts from arma::vec to np.array
%typemap(out, fragment="NumPy_Fragments") arma::vec {
    double* data = $1.memptr();
    npy_intp size[1] = {$1.n_elem};
    PyObject* array = PyArray_EMPTY( 1, size, NPY_DOUBLE, true);
    std::copy( data, data+size[0], reinterpret_cast<double*>(array_data(array)));
    return array;
}

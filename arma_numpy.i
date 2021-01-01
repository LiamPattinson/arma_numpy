// arma_numpy.i
// A SWIG interface file to convert between Armadillo vectors/matrices and Numpy arrays.

%header %{
#include <algorithm>
#include <type_traits>
#include <complex>
#include <armadillo>
#include <numpy/arrayobject.h>
%}

%include "numpy.i"
%include "std_complex.i"

%init %{
    import_array();
%}

// Define some new typechecking precedence values
// UINT64_ARRAY is INT64_ARRAY+1
// COMPLEX_FLOAT_ARRAY is DOUBLE_ARRAY+1
// COMPLEX_DOUBLE_ARRAY is DOUBLE_ARRAY+2
// Hopefully the swig devs won't invalidate this anytime soon!

%define SWIG_TYPECHECK_UINT64_ARRAY         1056 %enddef
%define SWIG_TYPECHECK_COMPLEX_FLOAT_ARRAY  1091 %enddef
%define SWIG_TYPECHECK_COMPLEX_DOUBLE_ARRAY 1092 %enddef

// Define some helpers for converting types to/from typecodes or getting info from a type

%fragment("arma_numpy_utilities","header"){

#include <armadillo>

    // type to numpy typecode
    template<class T> struct to_typecode {};
    template<> struct to_typecode<long long>            { static const int value = NPY_INT64; };
    template<> struct to_typecode<unsigned long long>   { static const int value = NPY_UINT64; };
    template<> struct to_typecode<float>                { static const int value = NPY_FLOAT; };
    template<> struct to_typecode<double>               { static const int value = NPY_DOUBLE; };
    template<> struct to_typecode<std::complex<float>>  { static const int value = NPY_COMPLEX64; };
    template<> struct to_typecode<std::complex<double>> { static const int value = NPY_COMPLEX128; };

    // numpy typecode to type
    template<int TYPE> struct from_typecode {};
    template<> struct from_typecode<NPY_INT64>      {  using type = long long; };
    template<> struct from_typecode<NPY_UINT64>     {  using type = unsigned long long; };
    template<> struct from_typecode<NPY_FLOAT>      {  using type = float; };
    template<> struct from_typecode<NPY_DOUBLE>     {  using type = double; };
    template<> struct from_typecode<NPY_COMPLEX64>  {  using type = arma::cx_float; };
    template<> struct from_typecode<NPY_COMPLEX128> {  using type = arma::cx_double; };

    // info from arma type
    template<class T> struct arma_info {
        // element type
        using element_t = typename T::elem_type;
        // corresponding typecode
        static constexpr int typecode = to_typecode<element_t>::value;
        // how many dims?
        static constexpr int dims = ( std::is_same<T,arma::Col<element_t>>::value || std::is_same<T,arma::Row<element_t>>::value ? 1 :
                                        ( std::is_same<T,arma::Mat<element_t>>::value ? 2 :
                                            ( std::is_same<T,arma::Cube<element_t>>::value ? 3 : 0 )
                                        )
                                    );
        static_assert( dims, "Cannot wrap Armadillo objects with more dims than Cube");
    };

}

// Define methods for converting np.array <-> 1D arma

%fragment("arma_numpy_1d", "header", fragment="NumPy_Fragments", fragment="arma_numpy_utilities"){

    /* Typecheck to ensure the input array is of the correct type and shape.
     * If a numpy array is not contiguous or is not column-ordered, a new array will be constructed for the conversion.
     * Element type is checked rigorously, i.e. dtype=='float32' maps to Col<float>, dtype=='float64' maps to Col<double>.
     * This may change in later updates.
     */
    template<class T>
    int arma_numpy_typecheck(PyObject* input){
        constexpr int typecode = arma_info<T>::typecode;
        constexpr int dims = arma_info<T>::dims;

        // Check input type is a numpy array
        if( !is_array(input) ) return false;

        // Convert to PyArrayObject for further testing
        PyArrayObject* array = (PyArrayObject*)input;
        int input_typecode = array_type(array);
        int input_dims = array_numdims(array);

        // Check numpy array has correct type
        if( !PyArray_EquivTypenums( input_typecode, typecode) ) return false;

        // Check numpy array has correct ndims
        if( input_dims != dims ) return false;

        return true;
    }
    
    /* Converts numpy array to arma::vec, or any variation on
     * the theme (e.g. Col<int>, frowvec, etc).
     * If the parameter 'copy' is set to false,
     * the new arma::vec will make use of the same memory as
     * the numpy array. Be warned that this can cause havoc
     * if you change the elements of the array or resize it
     * from c++.
     */
    template<class Vec>
    Vec numpy_to_arma_1d(PyObject* input, bool copy=true){
        // Determine internal type of Vec and corresponding numpy typecode
        using element_t = typename arma_info<Vec>::element_t;
        static constexpr int typecode = arma_info<Vec>::typecode;
        // Convert generic PyObject to Numpy array
        int is_new_object;
        PyArrayObject* array = obj_to_array_fortran_allow_conversion( input, typecode, &is_new_object);
        // Build new Vec using pointer to numpy array data
        // Will copy unless
        element_t* data = reinterpret_cast<element_t*>(array_data(array));
        arma::uword n_elem = static_cast<arma::uword>(array_dimensions(array)[0]);
        Vec v = Vec(data,n_elem,copy);
        return v;
    }

    /* Converts arma::vec to numpy arrays
     * This will currently always copy, but this may change in future versions
     */
    template<class Vec>
    PyObject* arma_to_numpy_1d( const Vec& v){
        // Get element type and corresponding type code
        using element_t = typename arma_info<Vec>::element_t;
        static constexpr int typecode = arma_info<Vec>::typecode;
        // Get size of vector, express it in a form numpy understands
        arma::uword size = v.n_elem;
        npy_intp dims[1] = {(npy_intp)size};
        // Create new empty array, copy elements over
        PyObject* array = PyArray_EMPTY( 1, dims, typecode, true);
        std::copy( v.begin(), v.end(), reinterpret_cast<element_t*>(array_data(array)));
        return array;
    }
}

// Create typemaps for armadillo typedefs
// (assumes ARMA_64BIT_WORD)
%apply long long { arma::sword};
%apply long long unsigned { arma::uword};
%apply std::complex<float> { arma::cx_float};
%apply std::complex<double> { arma::cx_double};

// Create macro for arma::vec typemaps
%define %gen_typemaps(Vec,prec)

    %typemap(in,  fragment="arma_numpy_1d") Vec, const Vec { $1 = numpy_to_arma_1d<Vec>($input); }
    %typemap(out, fragment="arma_numpy_1d") Vec, const Vec { $result = arma_to_numpy_1d<Vec>($1); }
    %typemap(typecheck, precedence=prec, fragment="arma_numpy_1d") Vec, const Vec { $1 = arma_numpy_typecheck<Vec>($input); }

%enddef

// Generate typemaps for arma::Col

%gen_typemaps(arma::ivec,SWIG_TYPECHECK_INT64_ARRAY);
%gen_typemaps(arma::uvec,SWIG_TYPECHECK_UINT64_ARRAY);
%gen_typemaps(arma::fvec,SWIG_TYPECHECK_FLOAT_ARRAY);
%gen_typemaps(arma::dvec,SWIG_TYPECHECK_DOUBLE_ARRAY);
%gen_typemaps(arma::cx_fvec,SWIG_TYPECHECK_COMPLEX_FLOAT_ARRAY);
%gen_typemaps(arma::cx_dvec,SWIG_TYPECHECK_COMPLEX_DOUBLE_ARRAY);

%apply arma::ivec { arma::icolvec, arma::Col<long long>, arma::Col<arma::sword>};
%apply arma::uvec { arma::ucolvec, arma::Col<unsigned long long>, arma::Col<arma::uword>};
%apply arma::fvec { arma::fcolvec, arma::Col<float>};
%apply arma::dvec { arma::dcolvec, arma::colvec, arma::vec, arma::Col<double>};
%apply arma::cx_fvec { arma::cx_fcolvec, arma::Col<std::complex<float>>};
%apply arma::cx_dvec { arma::cx_dcolvec, arma::cx_vec, arma::cx_colvec, arma::Col<std::complex<double>>};

// Generate typemaps for arma::Row

%gen_typemaps(arma::irowvec,SWIG_TYPECHECK_INT64_ARRAY);
%gen_typemaps(arma::urowvec,SWIG_TYPECHECK_UINT64_ARRAY);
%gen_typemaps(arma::frowvec,SWIG_TYPECHECK_FLOAT_ARRAY);
%gen_typemaps(arma::drowvec,SWIG_TYPECHECK_DOUBLE_ARRAY);
%gen_typemaps(arma::cx_frowvec,SWIG_TYPECHECK_COMPLEX_FLOAT_ARRAY);
%gen_typemaps(arma::cx_drowvec,SWIG_TYPECHECK_COMPLEX_DOUBLE_ARRAY);

%apply arma::irowvec { arma::Row<long long>, arma::Row<arma::sword>};
%apply arma::urowvec { arma::Row<unsigned long long>, arma::Row<arma::uword>};
%apply arma::frowvec { arma::Row<float>};
%apply arma::drowvec { arma::rowvec, arma::Row<double>};
%apply arma::cx_frowvec { arma::Row<std::complex<float>>};
%apply arma::cx_drowvec { arma::cx_rowvec, arma::Row<std::complex<double>>};


%typemap(out, fragment="NumPy_Fragments") arma::ivec {
    arma::uword size = $1.n_elem;
    npy_intp dims[1] = {(npy_intp)size};
    PyObject* array = PyArray_EMPTY( 1, dims, NPY_INT64, true);
    std::copy( $1.begin(), $1.end(), reinterpret_cast<arma::sword*>(array_data(array)));
    return array;
}

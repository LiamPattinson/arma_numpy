// arma_numpy.i
// A SWIG interface file to convert between Armadillo vectors/matrices and Numpy arrays.

%header %{
#include <array>
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
// UINT64_ARRAY is INT64_ARRAY-1
// COMPLEX_FLOAT_ARRAY is DOUBLE_ARRAY+1
// COMPLEX_DOUBLE_ARRAY is DOUBLE_ARRAY+2
// Hopefully the swig devs won't invalidate this anytime soon!

%define SWIG_TYPECHECK_UINT64_ARRAY         1054 %enddef
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

    // build arma object given memptr and dims
    template<class T, typename std::enable_if<arma_info<T>::dims==1,bool>::type = true>
    T arma_from_ptr( typename arma_info<T>::element_t* data, npy_intp* dims, bool copy){
        return T(data,dims[0],copy);
    }

    template<class T, typename std::enable_if<arma_info<T>::dims==2,bool>::type = true>
    T arma_from_ptr( typename arma_info<T>::element_t* data, npy_intp* dims, bool copy){
        return T(data,dims[0],dims[1],copy);
    }

    template<class T, typename std::enable_if<arma_info<T>::dims==3,bool>::type = true>
    T arma_from_ptr( typename arma_info<T>::element_t* data, npy_intp* dims, bool copy){
        return T(data,dims[0],dims[1],dims[2],copy);
    }

    // get shape of arma object
    template<class T, typename std::enable_if<arma_info<T>::dims==1,bool>::type = true>
    auto arma_shape( const T& t) -> typename std::array<npy_intp,arma_info<T>::dims> {
        return std::array<npy_intp,arma_info<T>::dims>{ (npy_intp)t.n_elem};
    }

    template<class T, typename std::enable_if<arma_info<T>::dims==2,bool>::type = true>
    auto arma_shape( const T& t) -> typename std::array<npy_intp,arma_info<T>::dims> {
        return std::array<npy_intp,arma_info<T>::dims>{ (npy_intp)t.n_rows, (npy_intp)t.n_cols};
    }

    template<class T, typename std::enable_if<arma_info<T>::dims==3,bool>::type = true>
    auto arma_shape( const T& t) -> typename std::array<npy_intp,arma_info<T>::dims> {
        return std::array<npy_intp,arma_info<T>::dims>{ (npy_intp)t.n_rows, (npy_intp)t.n_cols, (npy_intp)t.n_slices};
    }
}

// Define methods for converting between np.array and arma

%fragment("arma_numpy", "header", fragment="NumPy_Fragments", fragment="arma_numpy_utilities"){

    /* Typecheck to ensure the input array is of the correct type and shape.
     * If a numpy array is not contiguous or is not column-ordered, a new array will be constructed for the conversion.
     * Numpy arrays may be cast to a different type, though only if it can be done so safely.
     * Setting strict=true prevents casting, and imposes additional restrictions on internal memory layout.
     */
    template<class T>
    int arma_numpy_typecheck(PyObject* input, bool strict=false){
        constexpr int typecode = arma_info<T>::typecode;
        constexpr int dims = arma_info<T>::dims;

        // Check input type is a numpy array
        if( !is_array(input) ) return false;

        // Convert to PyArrayObject for further testing
        PyArrayObject* array = (PyArrayObject*)input;
        int input_typecode = array_type(array);
        int input_dims = array_numdims(array);

        // Check numpy array has correct type
        // (or it can be cast to it without losing information)
        if( !PyArray_CanCastSafely( input_typecode, typecode) ) return false;

        // Check numpy array has correct ndims
        if( input_dims != dims ) return false;

        // Apply stricter measures
        // * no type casting
        // * array must be column-ordered contiguous (e.g. Fortran style)
        // * array must own its own data (e.g. not a view)
        if( strict ){
            // Require equivalent typecodes (e.g. okay if NPY_INT and NPY_LONGINT are the same)
            if( !PyArray_EquivTypenums( input_typecode, typecode) ) return false;
            // Require column ordered (or at least contiguous if the array is 1D)
            if( !PyArray_IS_F_CONTIGUOUS(array) && !(input_dims == 1 && PyArray_IS_C_CONTIGUOUS(array)) ) return false;
           // Must own data
            if( !PyArray_CHKFLAGS(array,NPY_ARRAY_OWNDATA)) return false; 
        }

        return true;
    }
    
    /* Converts numpy array to arma container.
     * If the parameter 'copy' is set to false,
     * the new container will make use of the same memory as
     * the numpy array. Be warned that this can cause havoc
     * if you change the elements of the array or resize it
     * from c++.
     */
    template<class T>
    T numpy_to_arma(PyObject* input, bool copy=true){
        // Determine internal type of Vec and corresponding numpy typecode
        using element_t = typename arma_info<T>::element_t;
        static constexpr int typecode = arma_info<T>::typecode;
        // Convert generic PyObject to Numpy array
        PyArrayObject* array = NULL;
        if( copy ){
            int is_new_object;
            array = obj_to_array_fortran_allow_conversion( input, typecode, &is_new_object);
        } else {
            array = obj_to_array_no_conversion( input, typecode);
        }
        // Get dimensionality of numpy array and pointer to its raw data
        npy_intp* dims = array_dimensions(array);
        element_t* data = reinterpret_cast<element_t*>(array_data(array));
        T t = arma_from_ptr<T>(data,dims,copy);
        return t;
    }

    /* Converts arma container to numpy array
     * Returns by value, and always creates a new numpy array.
     */
    template<class T>
    PyObject* arma_to_numpy( const T& t ){
        // Get element type and corresponding type code
        using element_t = typename arma_info<T>::element_t;
        static constexpr int typecode = arma_info<T>::typecode;
        static constexpr int dims = arma_info<T>::dims;
        // Get shape of arma container
        std::array<npy_intp,dims> shape = arma_shape(t);
        // Create new empty array, copy elements over
        PyObject* array = PyArray_EMPTY( dims, shape.data(), typecode, /*'fortran' ordering*/ true);
        std::copy( t.begin(), t.end(), reinterpret_cast<element_t*>(array_data(array)));
        return array;
    }
}

// Create typemaps for armadillo typedefs
// (assumes ARMA_64BIT_WORD)
%apply long long { arma::sword};
%apply long long unsigned { arma::uword};
%apply std::complex<float> { arma::cx_float};
%apply std::complex<double> { arma::cx_double};

// Create macro for arma container typemaps
%define %gen_typemaps(T,prec)
    %typemap(in,  fragment="arma_numpy") T, const T  { $1 = numpy_to_arma<T>($input); }
    %typemap(in,  fragment="arma_numpy") T& (T temp), T* (T temp) { temp = numpy_to_arma<T>($input,false); $1 = &temp; }
    %typemap(in,  fragment="arma_numpy") const T& (T temp), const T* (T temp) { temp = numpy_to_arma<T>($input); $1 = &temp; }
    %typemap(out, optimal="1", fragment="arma_numpy") T { $result = arma_to_numpy<T>($1); }
    %typemap(typecheck, precedence=prec, fragment="arma_numpy") T, const T, T *, const T *, T &, const T & { $1 = arma_numpy_typecheck<T>($input); }
%enddef

// Some preprocessor magic...
#define GET_NTH_ARG( _1, _2, _3, _4, N, ...) N
#define APPLY_SYMBOLS_1(_pre,_post,_a) (_pre _a _post)
#define APPLY_SYMBOLS_2(_pre,_post,_a,...) (_pre _a _post) , APPLY_SYMBOLS_1(_pre,_post,__VA_ARGS__)
#define APPLY_SYMBOLS_3(_pre,_post,_a,...) (_pre _a _post) , APPLY_SYMBOLS_2(_pre,_post,__VA_ARGS__)
#define APPLY_SYMBOLS_4(_pre,_post,_a,...) (_pre _a _post) , APPLY_SYMBOLS_3(_pre,_post,__VA_ARGS__)
#define APPLY_SYMBOLS(_pre,_post,...) \
    GET_NTH_ARG(__VA_ARGS__,APPLY_SYMBOLS_4,APPLY_SYMBOLS_3,APPLY_SYMBOLS_2,APPLY_SYMBOLS_1)(_pre,_post,##__VA_ARGS__)

// Create macro to apply typemaps to all variations on the type T (e.g. vec, colvec*, const dvec&, dcolvec&, const Col<double>)
%define %apply_typemaps(T,...)
    %apply T  { APPLY_SYMBOLS(,,__VA_ARGS__) };
    %apply T& { APPLY_SYMBOLS(,&,__VA_ARGS__) };
    %apply T* { APPLY_SYMBOLS(,*,__VA_ARGS__) };
    %apply const T  { APPLY_SYMBOLS(const,,__VA_ARGS__) };
    %apply const T& { APPLY_SYMBOLS(const,&,__VA_ARGS__) };
    %apply const T* { APPLY_SYMBOLS(const,*,__VA_ARGS__) };
%enddef

// Generate typemaps for arma::Col

%gen_typemaps(arma::ivec,SWIG_TYPECHECK_INT64_ARRAY);
%gen_typemaps(arma::uvec,SWIG_TYPECHECK_UINT64_ARRAY);
%gen_typemaps(arma::fvec,SWIG_TYPECHECK_FLOAT_ARRAY);
%gen_typemaps(arma::dvec,SWIG_TYPECHECK_DOUBLE_ARRAY);
%gen_typemaps(arma::cx_fvec,SWIG_TYPECHECK_COMPLEX_FLOAT_ARRAY);
%gen_typemaps(arma::cx_dvec,SWIG_TYPECHECK_COMPLEX_DOUBLE_ARRAY);

%apply_typemaps( arma::ivec, arma::icolvec, arma::Col<long long>, arma::Col<arma::sword>);
%apply_typemaps( arma::uvec, arma::ucolvec, arma::Col<unsigned long long>, arma::Col<arma::uword>);
%apply_typemaps( arma::fvec, arma::fcolvec, arma::Col<float>);
%apply_typemaps( arma::dvec, arma::dcolvec, arma::colvec, arma::vec, arma::Col<double>);
%apply_typemaps( arma::cx_fvec, arma::cx_fcolvec, arma::Col<std::complex<float>>);
%apply_typemaps( arma::cx_dvec, arma::cx_dcolvec, arma::cx_vec, arma::cx_colvec, arma::Col<std::complex<double>>);

// Generate typemaps for arma::Row

%gen_typemaps(arma::irowvec,SWIG_TYPECHECK_INT64_ARRAY);
%gen_typemaps(arma::urowvec,SWIG_TYPECHECK_UINT64_ARRAY);
%gen_typemaps(arma::frowvec,SWIG_TYPECHECK_FLOAT_ARRAY);
%gen_typemaps(arma::drowvec,SWIG_TYPECHECK_DOUBLE_ARRAY);
%gen_typemaps(arma::cx_frowvec,SWIG_TYPECHECK_COMPLEX_FLOAT_ARRAY);
%gen_typemaps(arma::cx_drowvec,SWIG_TYPECHECK_COMPLEX_DOUBLE_ARRAY);

%apply_typemaps( arma::irowvec, arma::Row<long long>, arma::Row<arma::sword>);
%apply_typemaps( arma::urowvec, arma::Row<unsigned long long>, arma::Row<arma::uword>);
%apply_typemaps( arma::frowvec, arma::Row<float>);
%apply_typemaps( arma::drowvec, arma::rowvec, arma::Row<double>);
%apply_typemaps( arma::cx_frowvec, arma::Row<std::complex<float>>);
%apply_typemaps( arma::cx_drowvec, arma::cx_rowvec, arma::Row<std::complex<double>>);

// Generate typemaps for arma::Mat

%gen_typemaps(arma::imat,SWIG_TYPECHECK_INT64_ARRAY);
%gen_typemaps(arma::umat,SWIG_TYPECHECK_UINT64_ARRAY);
%gen_typemaps(arma::fmat,SWIG_TYPECHECK_FLOAT_ARRAY);
%gen_typemaps(arma::dmat,SWIG_TYPECHECK_DOUBLE_ARRAY);
%gen_typemaps(arma::cx_fmat,SWIG_TYPECHECK_COMPLEX_FLOAT_ARRAY);
%gen_typemaps(arma::cx_dmat,SWIG_TYPECHECK_COMPLEX_DOUBLE_ARRAY);

%apply_typemaps( arma::imat, arma::Mat<long long>, arma::Mat<arma::sword>);
%apply_typemaps( arma::umat, arma::Mat<unsigned long long>, arma::Mat<arma::uword>);
%apply_typemaps( arma::fmat, arma::Mat<float>);
%apply_typemaps( arma::dmat, arma::mat, arma::Mat<double>);
%apply_typemaps( arma::cx_fmat, arma::Mat<std::complex<float>>);
%apply_typemaps( arma::cx_dmat, arma::cx_mat, arma::Mat<std::complex<double>>);

// Generate typemaps for arma::Cube

%gen_typemaps(arma::icube,SWIG_TYPECHECK_INT64_ARRAY);
%gen_typemaps(arma::ucube,SWIG_TYPECHECK_UINT64_ARRAY);
%gen_typemaps(arma::fcube,SWIG_TYPECHECK_FLOAT_ARRAY);
%gen_typemaps(arma::dcube,SWIG_TYPECHECK_DOUBLE_ARRAY);
%gen_typemaps(arma::cx_fcube,SWIG_TYPECHECK_COMPLEX_FLOAT_ARRAY);
%gen_typemaps(arma::cx_dcube,SWIG_TYPECHECK_COMPLEX_DOUBLE_ARRAY);

%apply_typemaps( arma::icube, arma::Cube<long long>, arma::Cube<arma::sword>);
%apply_typemaps( arma::ucube, arma::Cube<unsigned long long>, arma::Cube<arma::uword>);
%apply_typemaps( arma::fcube, arma::Cube<float>);
%apply_typemaps( arma::dcube, arma::cube, arma::Cube<double>);
%apply_typemaps( arma::cx_fcube, arma::Cube<std::complex<float>>);
%apply_typemaps( arma::cx_dcube, arma::cx_cube, arma::Cube<std::complex<double>>);

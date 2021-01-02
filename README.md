# arma_numpy

SWIG interface to automatically convert between Armadillo vectors/matrices (C++) and Numpy arrays (Python).

## Installation

Simply copy the files `numpy.i` and  `arma_numpy.i` into your SWIG project, and
remember to `%include` them in your interface file. You may also need to ensure
that your Numpy `core/include/` directory is on your compiler's include path.

This interface has been designed to work alongside 64-bit versions of Numpy and
Armadillo, and it requires the C++11 standard. Your mileage may vary on 32-bit
machines, or when SWIG-ing with older versions of either library.

## Features

This interface permits seamless conversions between Numpy arrays and various
Armadillo container types. For example, consider the following (rather silly) 
C++ function:

```
arma::mat promote_to_double_and_transpose(arma::fmat mat);
```

We may wrap this function using SWIG, and call it from within Python using
Numpy arrays:

```
>>> import numpy as np
>>> x = np.linspace(0.,5.,6).reshape((2,3)).astype('float32')
>>> y = promote_to_double_and_transpose(x)
>>> y.dtype == 'float64'
True
>>> y.shape() == (3,2)
True
>>> np.all(y == x.T)
True
```

These conversions will happen automatically, with no further effort required
by the user. Typemaps have been defined to handle const and
non-const variants of pass-by-value, pass-by-reference, and pass-by-pointer.
Armadillo objects may also be returned by value and automatically converted
into Numpy arrays.
See the `./tests` folder for more examples.

Efforts have been made to avoid any unnecessary copying when converting between
Numpy and Armadillo types, and pointers to the raw underlying data are used
wherever it is safe to do so. In particular, passing 1D arrays by pointer
or reference is unlikely to require any copying. If it is necessary to convert
a Numpy array to a suitable format before passing to a C++ function, this
will be performed quietly behind-the-scenes.

Some cases exist where it is not generally possible to safely convert a
Numpy array; these are the topics of the next two sections.

### Multidimensional Arrays and Pass-by-Reference

When passing by non-const reference or pointer, we may edit Numpy arrays in-place via
Armadillo types. For example, take the following function:

```
void set_to_zeros( arma::imat& m){
    m.zeros();
}
```

To call this from Python, we first need to
call the function `np.asfortranarray` on any Numpy arrays we wish to pass in:

```
>>> x = np.linspace(0,8,9).reshape((3,3))
>>> x = np.asfortranarray(x)
>>> np.any(x)
True
>>> set_to_zeros(x)
>>> np.any(x)
False
```

Failing to include the explicit conversion to a Fortran array causes an error,
as SWIG is unable to find a matching typemap. The code would also fail if
the Numpy array's dtype did not match the element type of `arma::imat`.

These constraints are to ensure that Numpy arrays passed by non-const reference have
exactly the same underlying representation in memory as their corresponding
Armadillo types. By default, Numpy stores
multidimensional arrays in *row-major order* (C-style), meaning that that
if we call:

```
>>> A = np.linspace(0,8,9).reshape((3,3))
```

The matrix `A` is stored in memory as the following 1D array:

```
A_numpy_data -> { 0, 1, 2, 3, 4, 5, 6, 7, 8 }
```

Numpy may also try to be clever and store data in a non-contiguous manner, or
reference data belonging to a different Numpy array. All of these cases are
incompatible with Armadillo, as it prefers to store multidimensional arrays
contiguously and in *column-major order* (Fortran-style), meaning the underlying
representation is instead:

```
A_arma_data -> { 0, 3, 6, 1, 4, 7, 2, 5, 8 }
```

(Horrifying, I know.)

As we expect Armadillo containers passed by non-const reference may be edited in-place,
we require any corresponding Numpy arrays to have the same underlying representation.
This is not required of any Armadillo containers passed by value or const reference/pointer,
for which `arma_numpy.i` will instead quietly convert any incompatible inputs.

Note that conversions require copying the elements of the input array, so if you're
worried about speed, try to ensure any input arrays are Fortran ordered beforehand.
If a function still won't accept a Numpy array, check the array's flags:

```
>>> x.flags
    ...
    F_CONTIGUOUS : TRUE
    OWNDATA : TRUE
    WRITEABLE : TRUE
    ALIGNED : TRUE
    ...
```

If any of the above flags aren't `TRUE`, try making a copy using `np.asfortranarray`.

### Return by Reference

It is not possible to wrap functions that return Armadillo containers by reference. For
example, say you tried to wrap the function:

```
arma::vec* get_arma_vec_ptr(/* ...args... */);
```

In general, it is not possible to know what the returned pointer is actually pointing to.
Did this function allocate memory using `malloc` or `new`? Does any other part of the code
reference the same `arma::vec`? Could we later find this pointer `free`'d or `delete`'d
from elsewhere in the code? Though it is entirely possible to wrap this pointer
with a Numpy array and have Python manage its lifetime, I consider this
to be an unsafe operation, and hence these options have been omitted. This may change
in a future release.

### Other Limitations

It is not possible to make use of `arma::mat` (or other matrix types) as a catch-all for
any Numpy array. In Armadillo, it is permitted to pass vectors in function calls where
matrices are expected. When passing Numpy arrays, we expect that a function taking
a `arma::Col` or `arma::Row` type will take only 1D Numpy arrays, a function taking a
`arma::Mat` type will take only 2D Numpy arrays, etc.

Furthermore, there currently exists no option to wrap `arma::field` objects, nor is there
an option to wrap sparse matrices.

## Acknowledgements

Inspired and heavily influenced by [ArmaNpy](https://sourceforge.net/p/armanpy/wiki/Home).


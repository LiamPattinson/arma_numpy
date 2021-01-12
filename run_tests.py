import numpy as np
from arma_numpy_test import *

import pytest

class TestClass:
    def init(self):
        """
        Defines a collection of numpy arrays of various shape, type, and order.
        Must be defined as a regular function rather than __init__ and called manually
        in each test, as otherwise pytest complains.
        """
        self.i32_1 = np.linspace(0,2,3).astype('int32')
        self.fi32_1 = np.asfortranarray(self.i32_1)
        self.i64_1 = np.linspace(0,2,3).astype('int64')
        self.fi64_1 = np.asfortranarray(self.i64_1)

        self.f32_1 = np.linspace(0,2,3).astype('float32')
        self.ff32_1 = np.asfortranarray(self.f32_1)
        self.f64_1 = np.linspace(0,2,3).astype('float64')
        self.ff64_1 = np.asfortranarray(self.f64_1)

        self.i64_2 = np.linspace(0,8,9).reshape((3,3)).astype('int64')
        self.fi64_2 = np.asfortranarray(self.i64_2)
        self.f64_2 = np.linspace(0,8,9).reshape((3,3)).astype('float64')
        self.ff64_2 = np.asfortranarray(self.f64_2)

        self.i64_3 = np.linspace(0,26,27).reshape((3,3,3)).astype('int64')
        self.fi64_3 = np.asfortranarray(self.i64_3)
        self.f64_3 = np.linspace(0,26,27).reshape((3,3,3)).astype('float64')
        self.ff64_3 = np.asfortranarray(self.f64_3)

    def test_sum_vec(self):
        """
        As 1D Numpy arrays are F-contiguous and C-contiguous, we should
        expect this to work on all variants of vector.
        Arrays containing float32 and int32 should be automatically upcast
        to float64 and int64.
        """
        self.init()
        assert sum_vec(self.i32_1) == np.sum(self.i32_1)
        assert sum_vec(self.fi32_1) == np.sum(self.i32_1)
        assert sum_vec(self.i64_1) == np.sum(self.i64_1)
        assert sum_vec(self.fi64_1) == np.sum(self.i64_1)
        assert sum_vec(self.f32_1) == np.sum(self.f32_1)
        assert sum_vec(self.ff32_1) == np.sum(self.ff32_1)
        assert sum_vec(self.f64_1) == np.sum(self.f64_1)
        assert sum_vec(self.ff64_1) == np.sum(self.ff64_1)

    def test_get_vec(self):
        """
        Ensures that return-by-value is working correctly.
        """
        x = get_vec(5)
        assert x.shape == (5,)
        assert np.all(x == np.zeros(5))

    def test_reverse_vec(self):
        """
        Test both input and output at the same time. Again, expect this to work for all vector types.
        Also expect to this upcast 32 bit types to 64 bits.
        Finally, try inputting an array with uint32 type. Expect it to upcast to uint64, not int64.
        """
        self.init()
        assert np.all(reverse_vec(self.i32_1) == self.i32_1[::-1])
        assert np.all(reverse_vec(self.fi32_1) == self.fi32_1[::-1])
        assert np.all(reverse_vec(self.i64_1) == self.i64_1[::-1])
        assert np.all(reverse_vec(self.fi64_1) == self.fi64_1[::-1])
        assert np.all(reverse_vec(self.f32_1) == self.f32_1[::-1])
        assert np.all(reverse_vec(self.ff32_1) == self.fi32_1[::-1])
        assert np.all(reverse_vec(self.f64_1) == self.f64_1[::-1])
        assert np.all(reverse_vec(self.ff64_1) == self.ff64_1[::-1])
        assert reverse_vec(self.i32_1).dtype == 'int64'
        assert reverse_vec(self.fi32_1).dtype == 'int64'
        assert reverse_vec(self.f32_1).dtype == 'float64'
        assert reverse_vec(self.ff32_1).dtype == 'float64'
        u = np.linspace(0,2,3).astype('uint32')
        assert np.all(reverse_vec(u) == u[::-1])
        assert reverse_vec(u).dtype == 'uint64'

    def test_set_vec_to_zero(self):
        """
        Expect numpy arrays to be modified inplace.
        Should work for all inputs (though some will run considerably slower as copies must be made for compatibility)
        """
        self.init()
        set_to_zero_by_ptr(self.f64_1)
        set_to_zero_by_ref(self.ff64_1)
        set_to_zero_by_ref(self.i32_1)
        set_to_zero_by_ref(self.fi32_1)
        set_to_zero_by_ref(self.i64_1)
        set_to_zero_by_ref(self.fi64_1)
        set_to_zero_by_ref(self.f32_1)
        assert np.all(self.f64_1 == 0.)
        assert np.all(self.ff64_1 == 0.)
        assert np.all(self.i32_1 == 0)
        assert np.all(self.fi32_1 == 0)
        assert np.all(self.i64_1 == 0)
        assert np.all(self.fi64_1 == 0)
        assert np.all(self.f32_1 == 0.)
        assert np.all(self.ff32_1 == 0.)

    def test_sum_mat(self):
        """
        Expect this to work on all mat types.
        As it takes input by value, a copy will be made if it doesn't match the exact type.
        """
        self.init()
        assert sum_mat(self.i64_2) == np.sum(self.i64_2)
        assert sum_mat(self.fi64_2) == np.sum(self.fi64_2)
        assert sum_mat(self.f64_2) == np.sum(self.f64_2)
        assert sum_mat(self.ff64_2) == np.sum(self.ff64_2)


    def test_transpose_mat(self):
        """
        Takes by value, returns by value. Expect it to work for all mat types.
        However, all returned arrays should be upcast to dtype=float64, and should be Fortran contiguous.
        """
        self.init()
        assert np.all(transpose_mat(self.i64_2) == self.i64_2.T)
        assert np.all(transpose_mat(self.fi64_2) == self.i64_2.T)
        assert np.all(transpose_mat(self.f64_2) == self.f64_2.T)
        assert np.all(transpose_mat(self.ff64_2) == self.ff64_2.T)
        assert transpose_mat(self.i64_2).dtype == 'float64'
        assert transpose_mat(self.fi64_2).dtype == 'float64'
        assert transpose_mat(self.f64_2).dtype == 'float64'
        assert transpose_mat(self.ff64_2).dtype == 'float64'
        assert transpose_mat(self.i64_2).flags['F_CONTIGUOUS'] == True
        assert transpose_mat(self.fi64_2).flags['F_CONTIGUOUS'] == True
        assert transpose_mat(self.f64_2).flags['F_CONTIGUOUS'] == True
        assert transpose_mat(self.ff64_2).flags['F_CONTIGUOUS'] == True
        assert transpose_mat(self.i64_2).flags['C_CONTIGUOUS'] == False
        assert transpose_mat(self.fi64_2).flags['C_CONTIGUOUS'] == False
        assert transpose_mat(self.f64_2).flags['C_CONTIGUOUS'] == False
        assert transpose_mat(self.ff64_2).flags['C_CONTIGUOUS'] == False

    def test_set_mat_to_zero(self):
        """
        Expect this to work for all.
        """
        self.init()
        set_to_zero_by_ref(self.ff64_2)
        assert np.all(self.ff64_2 == 0)
        # Set to ones, then test set_to_zero_by_ptr also works
        self.ff64_2 += 1
        assert np.all(self.ff64_2)
        set_to_zero_by_ptr(self.ff64_2)
        assert np.all(self.ff64_2 == 0)
        # Repeat for other matrix types
        # These ones will make internal copies, so will run slower.
        set_to_zero_by_ref(self.i64_2)
        set_to_zero_by_ref(self.fi64_2)
        set_to_zero_by_ref(self.f64_2)
        assert np.all(self.i64_2 == 0)
        assert np.all(self.fi64_2 == 0)
        assert np.all(self.f64_2 == 0)
        self.i64_2 += 1
        self.fi64_2 += 1
        self.f64_2 += 1
        assert np.all(self.i64_2)
        assert np.all(self.fi64_2)
        assert np.all(self.f64_2)
        set_to_zero_by_ptr(self.i64_2)
        set_to_zero_by_ptr(self.fi64_2)
        set_to_zero_by_ptr(self.f64_2)
        assert np.all(self.i64_2 == 0)
        assert np.all(self.fi64_2 == 0)
        assert np.all(self.f64_2 == 0)


    def test_sum_mat_by_const_ref(self):
        """
        Ensure that passing by const ref always works for all mat types.
        It should default to pass by reference, but resort to pass by value if it is the wrong type or order.
        As no int version is defined, the returned value should always be float.
        """
        self.init()
        print( np.sum(self.i64_2))
        print( sum_mat_by_const_ref(self.i64_2))
        assert sum_mat_by_const_ref(self.i64_2) == np.sum(self.i64_2)
        assert sum_mat_by_const_ref(self.fi64_2) == np.sum(self.fi64_2)
        assert sum_mat_by_const_ref(self.f64_2) == np.sum(self.f64_2)
        assert sum_mat_by_const_ref(self.ff64_2) == np.sum(self.ff64_2)
        assert sum_mat_by_const_ptr(self.i64_2) == np.sum(self.i64_2)
        assert sum_mat_by_const_ptr(self.fi64_2) == np.sum(self.fi64_2)
        assert sum_mat_by_const_ptr(self.f64_2) == np.sum(self.f64_2)
        assert sum_mat_by_const_ptr(self.ff64_2) == np.sum(self.ff64_2)
        assert type(sum_mat_by_const_ref(self.i64_2)) == float
        assert type(sum_mat_by_const_ref(self.fi64_2)) == float
        assert type(sum_mat_by_const_ref(self.f64_2)) == float
        assert type(sum_mat_by_const_ref(self.ff64_2)) == float

    def test_sum_cube(self):
        """
        Should work for all cube types (pass by value)
        Test that the int version returns an int, and float version returns a float
        """
        self.init()
        assert sum_cube(self.i64_3) == np.sum(self.i64_3)
        assert sum_cube(self.fi64_3) == np.sum(self.fi64_3)
        assert sum_cube(self.f64_3) == np.sum(self.f64_3)
        assert sum_cube(self.ff64_3) == np.sum(self.ff64_3)
        assert type(sum_cube(self.i64_3)) == int
        assert type(sum_cube(self.fi64_3)) == int
        assert type(sum_cube(self.f64_3)) == float
        assert type(sum_cube(self.ff64_3)) == float

    def test_do_nothing_cube(self):
        """
        Should take a cube, not modify it, and return the same cube.
        This should work for all cubes, but the returned cubes will have fortran order and have their types converted to float
        """
        self.init()
        assert np.all(do_nothing(self.i64_3) == self.i64_3)
        assert np.all(do_nothing(self.fi64_3) == self.fi64_3)
        assert np.all(do_nothing(self.f64_3) == self.f64_3)
        assert np.all(do_nothing(self.ff64_3) == self.ff64_3)
        assert do_nothing(self.i64_3).flags['F_CONTIGUOUS'] == True
        assert do_nothing(self.fi64_3).flags['F_CONTIGUOUS'] == True
        assert do_nothing(self.f64_3).flags['F_CONTIGUOUS'] == True
        assert do_nothing(self.ff64_3).flags['F_CONTIGUOUS'] == True
        assert do_nothing(self.i64_3).dtype == 'float64'
        assert do_nothing(self.fi64_3).dtype == 'float64'
        assert do_nothing(self.f64_3).dtype == 'float64'
        assert do_nothing(self.ff64_3).dtype == 'float64'

    def test_get_second_slice(self):
        """
        Simply tests taking in a 3D object and returning a 2D one.
        Again, the returned object should by float64 and f-contiguous
        """
        self.init()
        assert np.all(get_second_slice(self.i64_3) == self.i64_3[:,:,1])
        assert np.all(get_second_slice(self.fi64_3) == self.fi64_3[:,:,1])
        assert np.all(get_second_slice(self.f64_3) == self.f64_3[:,:,1])
        assert np.all(get_second_slice(self.ff64_3) == self.ff64_3[:,:,1])
        assert get_second_slice(self.i64_3).shape == (3,3)
        assert get_second_slice(self.fi64_3).shape == (3,3)
        assert get_second_slice(self.f64_3).shape == (3,3)
        assert get_second_slice(self.ff64_3).shape == (3,3)
        assert get_second_slice(self.i64_3).dtype == 'float64'
        assert get_second_slice(self.fi64_3).dtype == 'float64'
        assert get_second_slice(self.f64_3).dtype == 'float64'
        assert get_second_slice(self.ff64_3).dtype == 'float64'
        assert get_second_slice(self.i64_3).flags['F_CONTIGUOUS'] == True
        assert get_second_slice(self.fi64_3).flags['F_CONTIGUOUS'] == True
        assert get_second_slice(self.f64_3).flags['F_CONTIGUOUS'] == True
        assert get_second_slice(self.ff64_3).flags['F_CONTIGUOUS'] == True

    def test_upcast_dims(self):
        """
        It should be possible to cast vectors to matrices and matrices to cubes.
        """
        self.init()
        # Test sum
        assert sum_mat(self.i64_1) == np.sum(self.i64_1)
        assert sum_cube(self.f64_1) == np.sum(self.f64_1)
        assert sum_cube(self.fi64_2) == np.sum(self.fi64_2)
        # Test transpose
        assert self.i64_1.shape == (3,)
        assert transpose_mat(self.i64_1).shape == (1,3)
        assert np.all(transpose_mat(self.i64_1) == self.i64_1.T)
        # Test that downcasting is not possible
        with pytest.raises(TypeError):
            transpose_mat(self.i64_3)
        with pytest.raises(TypeError):
            sum_vec(self.ff64_2)

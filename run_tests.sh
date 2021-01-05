#!/bin/bash

ERR_FLAGS="-Wall -Wextra"
CXX_FLAGS="${ERR_FLAGS} -std=c++11 -fPIC -O2"
SWIG_FLAGS="-python -py3 -c++ -cppext cpp ${ERR_FLAGS}"

# Call swig to generate interface files
swig $SWIG_FLAGS arma_numpy_test.i

# Compile all
g++ $CXX_FLAGS -c arma_numpy_test.cpp
g++ $CXX_FLAGS -c arma_numpy_test_wrap.cpp -I/usr/include/python3.9/ -I/usr/lib/python3.9/site-packages/numpy/core/include
g++ $CXX_FLAGS -shared arma_numpy_test.o arma_numpy_test_wrap.o -o _arma_numpy_test.so

# Test
pytest run_tests.py -v

#!/bin/bash

ERR_FLAGS="-Wall -Wextra -DARMA_NUMPY_DEBUG"
CXX_FLAGS="${ERR_FLAGS} -std=c++11 -fPIC -O2"
SWIG_FLAGS="-python -py3 -c++ -cppext cpp ${ERR_FLAGS}"

# Call swig to generate interface files
swig $SWIG_FLAGS test.i

# Compile all
g++ $CXX_FLAGS -c test.cpp
g++ $CXX_FLAGS -c test_wrap.cpp -I/usr/include/python3.9/ -I/usr/lib/python3.9/site-packages/numpy/core/include
g++ $CXX_FLAGS -shared test.o test_wrap.o -o _test.so


cmake_minimum_required(VERSION 3.8)
project(arma_numpy)

# CMake settings for arma_numpy.
# Primarily creates test suite.

# Set requirements and dependencies

set(CMAKE_CXX_STANDARD 11)
set(CXX_STANDARD_REQUIRED True)

find_package(Python REQUIRED COMPONENTS Interpreter NumPy)
include_directories(${Python_INCLUDE_DIRS})
include_directories(${Python_NumPy_INCLUDE_DIRS})

find_package(SWIG REQUIRED COMPONENTS python)
include(${SWIG_USE_FILE})

# Set compile options

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -fPIC -g")
#set(CMAKE_SWIG_FLAGS "-python -py3 -c++ -cppext cpp -Wall -Wextra") # Apparently not needed!

include_directories("${CMAKE_SOURCE_DIR}/")
include_directories("${CMAKE_SOURCE_DIR}/test/")

# Compile tests

set(TEST_DIR "${CMAKE_SOURCE_DIR}/test")
set_source_files_properties( "${TEST_DIR}/arma_numpy_test.i" PROPERTIES CPLUSPLUS ON)
swig_add_library( arma_numpy_test
    TYPE SHARED
    LANGUAGE python 
    SOURCES "${TEST_DIR}/arma_numpy_test.i" "${TEST_DIR}/arma_numpy_test.cpp"
)

# Add tests (call 'ctest --verbose')

enable_testing()
add_test(NAME "arma_numpy_test.exe" COMMAND pytest -v "run_tests.py")

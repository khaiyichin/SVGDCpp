# SVGDCpp: Stein Variational Gradient Descent in C++

## Introduction
This library provides the [Stein Variational Gradient Descent](https://arxiv.org/abs/1608.04471) algorithm in C++. In brief, SVGD is a hybrid (variational + sampling) method that estimates an assumed model using particles. The original work proposed SVGD as a _general purpose Bayesian inference algorithm_.

## Requirements
- [Eigen 3.3+](https://eigen.tuxfamily.org/dox/index.html) - _can be installed using `apt install libeigen3-dev`_
- [CppAD v20240602+](https://github.com/coin-or/CppAD) - _from source; when installing, configure your CMake with the following flags:_
    - `-D cppad_testvector=eigen`
    - `-D include_eigen=true`
    - `-D CMAKE_BUILD_TYPE=Release`
- [OpenMP v4.5+ (201511)](https://www.openmp.org/) - _should be shipped with your compiler_
- [GraphViz](https://graphviz.org/) - _required only if documentation is desired; can be installed using `apt install graphviz`_
- [Doxygen](https://www.doxygen.nl/) - _required only if documentation is desired; can be installed using `apt install doxygen`_

## Installation
1. Clone this repository and enter it.
2. Create a build directory.
    ```
    $ mkdir build
    $ cd build
    ```
3. Configure build flags (the provided values are the default).
    ```
    $ cmake .. -D BUILD_EXAMPLES=FALSE \
        -D BUILD_DOCUMENTATION=FALSE \
        -D CMAKE_BUILD_TYPE=Release
    ```
4. Build and install.
    ```
    $ make -j
    $ make install # may require sudo privileges depending on your CMAKE_INSTALL_PREFIX
    ```

## Getting Started
See the [examples directory](./examples/) for tutorials on how to use them and see [here](./doc/instructions.md) for detailed instructions.

## Tests
Unit tests have been provided to aid source code development. Besides identifying the kinds of testing imposed on the source code, looking into the test files can help you understand how the algorithm works. All you need to do is build the code with `-D CMAKE_BUILD_TYPE=Debug` and then run the tests either using `CTest`:
```
# in the build/ directory
$ make test
```
or run them individually (the tests are written with the [`doctest`](https://github.com/doctest/doctest) framework):
```
$ tests/test_model -s
```
#ifndef SVGD_CPP_CORE_HPP
#define SVGD_CPP_CORE_HPP

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <cppad/cppad.hpp>
#include <exception>
#include <type_traits>

using VectorXADd = Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, 1>;
using MatrixXADd = Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, Eigen::Dynamic>;

#endif
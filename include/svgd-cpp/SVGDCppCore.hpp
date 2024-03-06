#ifndef SVGD_CPP_CORE_HPP
#define SVGD_CPP_CORE_HPP

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <cppad/cppad.hpp>
#include <exception>
#include <type_traits>
#include <memory>

using VectorXADd = Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, 1>;
using MatrixXADd = Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, Eigen::Dynamic>;

/**
 * @brief Check whether two vectors have the same number of rows
 *
 * @tparam T1 Type of first vector, should be either Eigen::VectorXd or VectorXADd
 * @tparam T2 Type of second vector, should be either Eigen::VectorXd or VectorXADd
 * @param a First vector
 * @param b Second vector
 * @return true Both vectors have same number of rows
 * @return false Vectors have different number of rows
 */
template <typename T1, typename T2>
bool CompareVectorSizes(const T1 &a, const T2 &b)
{
    return (a.rows() == b.rows());
}

template <typename T1, typename T2>
bool CompareMatrixSizes(const T1 &a, const T2 &b)
{
    return (a.rows() == b.rows()) && (a.cols() == b.cols());
}

#endif
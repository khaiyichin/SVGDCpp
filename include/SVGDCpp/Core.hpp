/**
 * @file Core.hpp
 * @author Khai Yi Chin (khaiyichin@gmail.com)
 * @brief Core header to provide common types and functions.
 * @version 0.1
 * @date 2024-03-23
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef SVGD_CPP_CORE_HPP
#define SVGD_CPP_CORE_HPP

#include <Eigen/Core>
#include <Eigen/LU>
#include <cppad/cppad.hpp>
#include <type_traits>
#include <memory>
#include <numeric>
#include <cmath>
#include <utility>

#include "Exceptions.hpp"

/**
 * @brief Alias for variable-sized Eigen vector of type CppAD::AD<double>.
 * @ingroup Core_Module
 */
using VectorXADd = Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, 1>;

/**
 * @brief Alias for variable-sized Eigen matrix of type CppAD::AD<double>.
 * @ingroup Core_Module
 */
using MatrixXADd = Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, Eigen::Dynamic>;

/**
 * @brief Check whether two vectors have the same number of rows.
 *
 * @tparam T1 Type of first vector, should be either Eigen::VectorXd or @ref VectorXADd.
 * @tparam T2 Type of second vector, should be either Eigen::VectorXd or @ref VectorXADd.
 * @param a First vector.
 * @param b Second vector.
 * @return True if both vectors have same number of rows, false otherwise.
 *
 * @ingroup Core_Module
 */
template <typename T1, typename T2>
bool CompareVectorSizes(const T1 &a, const T2 &b)
{
    return (a.rows() == b.rows());
}

/**
 * @brief Convert a regular matrix of type double to a CppAD::AD<double> type matrix.
 *
 * @param eigen_mat Dynamic matrix of type double.
 * @return The input matrix but of Cpp::AD<double> type.
 */
MatrixXADd ConvertToCppAD(const Eigen::MatrixXd &eigen_mat)
{
    return eigen_mat.unaryExpr([](const double &value)
                               { return CppAD::Var2Par(CppAD::AD<double>(value)); });
}

/**
 * @brief Convert a CppAD::AD<double> type matrix to a regular matrix of type double.
 *
 * @param cppad_mat Dynamic matrix of type CppAD::AD<double>.
 * @return The input matrix but of double type.
 */
Eigen::MatrixXd ConvertFromCppAD(const MatrixXADd &cppad_mat)
{
    return cppad_mat.unaryExpr([](const CppAD::AD<double> &value)
                               { return CppAD::Value(value); });
}

#endif
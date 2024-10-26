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

#ifndef SVGDCPP_CORE_HPP
#define SVGDCPP_CORE_HPP

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
inline bool CompareVectorSizes(const T1 &a, const T2 &b)
{
    return (a.rows() == b.rows());
}

/**
 * @brief Convert a regular matrix of type double to a CppAD::AD<double> type matrix.
 *
 * @param eigen_mat Dynamic matrix of type double.
 * @return The input matrix but of Cpp::AD<double> type.
 */
inline MatrixXADd ConvertToCppAD(const Eigen::MatrixXd &eigen_mat)
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
inline Eigen::MatrixXd ConvertFromCppAD(const MatrixXADd &cppad_mat)
{
    return cppad_mat.unaryExpr([](const CppAD::AD<double> &value)
                               { return CppAD::Value(value); });
}

/**
 * @brief Utility function to set up CppAD so that SVGD can run in parallel mode.
 * @details Inspired by https://github.com/coin-or/CppAD/issues/197#issuecomment-1983462984.
 *
 */
inline void SetupForParallelMode()
{
    // Define the helper functions within the scope of the inline function
    auto in_parallel = []() -> bool
    {
        return omp_in_parallel() != 0;
    };

    auto thread_number = []() -> size_t
    {
        return static_cast<size_t>(omp_get_thread_num());
    };

    // Setup for multi-threading environment with CppAD
    CppAD::thread_alloc::parallel_setup(omp_get_max_threads(), in_parallel, thread_number);
    CppAD::thread_alloc::hold_memory(true); // it should be faster, even when num_thread is equal to one,
                                            // for thread_alloc to hold onto memory.

    CppAD::parallel_ad<double>(); // Setup CppAD for parallel use with AD types

    // Verify that Eigen::VectorXd and VectorXADd meet the requirements of SimpleVector for CppAD
    CppAD::CheckSimpleVector<double, Eigen::VectorXd>();
    CppAD::CheckSimpleVector<CppAD::AD<double>, VectorXADd>();
}

#endif
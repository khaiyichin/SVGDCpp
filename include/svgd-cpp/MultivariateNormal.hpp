#ifndef MULTIVARIATE_NORMAL_HPP
#define MULTIVARIATE_NORMAL_HPP

#include "SVGDCppCore.hpp"

class MultivariateNormal
{
public:
    MultivariateNormal(const Eigen::VectorXd &mean, const Eigen::MatrixXd &covariance)
    {
        // Ensure that the dimensions of mean matches covariance
        if (!CompareVectorSizes<Eigen::VectorXd, Eigen::VectorXd>(mean, covariance.col(0)) ||
            !CompareVectorSizes<Eigen::VectorXd, Eigen::VectorXd>(mean, covariance.row(0)))
        {
            throw std::runtime_error("Dimensions of parameter vectors/matrices do not match.");
        }

        // Store a regular and CppAD copy
        mean_vec_ = mean;
        mean_vec_ad_ = mean.cast<CppAD::AD<double>>();
        cov_mat_ = covariance;
        cov_mat_ad_ = covariance.cast<CppAD::AD<double>>();
    }

    ~MultivariateNormal(){};

    template <typename T>
    T PDF(const T &x)
    {
        return NormConst() * Kernel<T>(x).array();
    }

    template <typename T>
    T LogPDF(const T &x)
    {
        return PDF<T>(x).array().log();
    }

protected:
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

    double NormConst()
    {
        return 1.0 / (std::pow(2.0 * M_PI, mean_vec_.rows() / 2.0) * std::sqrt(cov_mat_.determinant()));
    }

    template <typename T>
    T Kernel(const T &x)
    {
        // Compute and store kernel value into their respective container types
        if constexpr (std::is_same<T, VectorXADd>::value)
        {
            VectorXADd diff = x - mean_vec_ad_;
            return (-0.5 * (diff.transpose() * cov_mat_ad_.inverse() * diff).array()).exp();
        }
        else if constexpr (std::is_same<T, Eigen::VectorXd>::value)
        {
            Eigen::VectorXd diff = x - mean_vec_;
            return (-0.5 * (diff.transpose() * cov_mat_.inverse() * diff).array()).exp();
        }
        else
        {
            throw std::runtime_error("Unknown type for x. Must use either Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, 1> or Eigen::VectorXd.");
        }
    }

    Eigen::VectorXd mean_vec_;

    Eigen::MatrixXd cov_mat_;

    VectorXADd mean_vec_ad_;

    MatrixXADd cov_mat_ad_;
};

#endif
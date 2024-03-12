#ifndef MULTIVARIATE_NORMAL_HPP
#define MULTIVARIATE_NORMAL_HPP

#include "SVGDCppCore.hpp"
#include "Distribution.hpp"

class MultivariateNormal : public Distribution
{
public:
    MultivariateNormal() {}

    MultivariateNormal(const Eigen::VectorXd &mean, const Eigen::MatrixXd &covariance) : Distribution(mean.rows())
    {
        // Ensure that the dimensions of mean matches covariance
        if (!CompareVectorSizes<Eigen::VectorXd, Eigen::VectorXd>(mean, covariance.col(0)) ||
            !CompareVectorSizes<Eigen::VectorXd, Eigen::VectorXd>(mean, covariance.row(0)))
        {
            throw std::runtime_error("Dimensions of parameter vectors/matrices do not match.");
        }

        // Store a regular and CppAD copy
        mean_vec_ad_ = mean.cast<CppAD::AD<double>>();
        cov_mat_ad_ = covariance.cast<CppAD::AD<double>>();

        // Compute the normalization constant based on the updated parameters
        ComputeNormConst();
    }

    MultivariateNormal(const MultivariateNormal &obj)
    {
        *this = obj;
    }

    ~MultivariateNormal(){};

    MultivariateNormal &operator=(const MultivariateNormal &obj)
    {
        dimension_ = obj.dimension_;
        mean_vec_ad_ = obj.mean_vec_ad_;
        cov_mat_ad_ = obj.cov_mat_ad_;

        Distribution::operator=(obj);

        return *this;
    }

    void UpdateParameters(const std::vector<Eigen::MatrixXd> &params) override
    {
        Eigen::VectorXd mean = params[0];
        Eigen::MatrixXd covariance = params[1];

        // Ensure that the dimensions of mean matches covariance
        if (!CompareVectorSizes<Eigen::VectorXd, Eigen::VectorXd>(mean, covariance.col(0)) ||
            !CompareVectorSizes<Eigen::VectorXd, Eigen::VectorXd>(mean, covariance.row(0)))
        {
            throw std::runtime_error("Dimensions of parameter vectors/matrices do not match.");
        }

        mean_vec_ad_ = mean.cast<CppAD::AD<double>>();
        cov_mat_ad_ = covariance.cast<CppAD::AD<double>>();

        // Compute the normalization constant based on the updated parameters
        ComputeNormConst();
    }

    void Step() override {}

protected:
    VectorXADd Kernel(const VectorXADd &x) override
    {
        VectorXADd diff = x - mean_vec_ad_;
        return (-0.5 * (diff.transpose() * cov_mat_ad_.inverse() * diff).array()).exp();
    }

    void ComputeNormConst() override
    {
        norm_const_ = 1.0 /
                      (std::pow(2.0 * M_PI, dimension_ / 2.0) * std::sqrt(CppAD::Value(cov_mat_ad_.determinant())));
    }

    VectorXADd mean_vec_ad_;

    MatrixXADd cov_mat_ad_;
};

#endif
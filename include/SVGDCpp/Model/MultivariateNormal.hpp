#ifndef SVGD_CPP_MULTIVARIATE_NORMAL_HPP
#define SVGD_CPP_MULTIVARIATE_NORMAL_HPP

#include "../Core.hpp"
#include "Model.hpp"

class MultivariateNormal : public Model
{
public:
    MultivariateNormal() {}

    MultivariateNormal(const Eigen::VectorXd &mean, const Eigen::MatrixXd &covariance) : Model(mean.rows())
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
        ComputeNormalizationConstant();
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

        Model::operator=(obj);

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
        ComputeNormalizationConstant();
    }

    double EvaluateModelNormalized(const Eigen::VectorXd &x)
    {
        return norm_const_ * EvaluateModel(x);
    }

    double EvaluateLogModelNormalized(const Eigen::VectorXd &x)
    {
        return std::log(norm_const_) + EvaluateLogModel(x);
    }

    Eigen::VectorXd EvaluateModelGradNormalized(const Eigen::VectorXd &x)
    {
        return norm_const_ * model_fun_ad_.Jacobian(x);
    }

    double GetNormalizationConstant() {return norm_const_;}

protected:
    VectorXADd ModelFun(const VectorXADd &x) override
    {
        VectorXADd diff = x - mean_vec_ad_;
        return (-0.5 * (diff.transpose() * cov_mat_ad_.inverse() * diff).array()).exp();
    }

    void ComputeNormalizationConstant()
    {
        norm_const_ = 1.0 /
                      (std::pow(2.0 * M_PI, dimension_ / 2.0) * std::sqrt(CppAD::Value(cov_mat_ad_.determinant())));
    }

    VectorXADd mean_vec_ad_;

    MatrixXADd cov_mat_ad_;

    double norm_const_;
};

#endif
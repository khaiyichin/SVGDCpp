#ifndef SVGD_CPP_BINOMIAL_NORMAL_POSTERIOR_HPP
#define SVGD_CPP_BINOMIAL_NORMAL_POSTERIOR_HPP

#include "../Core.hpp"
#include "Model.hpp"

class BinomialNormalPosterior : public Model
{
public:
    BinomialNormalPosterior() {}

    ~BinomialNormalPosterior() {}

    BinomialNormalPosterior(const double &n,
                            const double &t,
                            const double &f,
                            const Eigen::VectorXd &mean,
                            const Eigen::MatrixXd &covariance)
    {
        // Check whether parameters are reasonable
        if (n > t)
        {
            throw std::runtime_error("SVGDCpp: Parameter n should be at most equal to parameter t.");
        }
        else if (n < 0 || t <= 0)
        {
            throw std::runtime_error("SVGDCpp: Parameter n should be > 0 and/or parameter t should be non-negative.");
        }

        model_parameters_[0] = (Eigen::Matrix<int, 1, 1>() << n).finished();
        model_parameters_[1] = (Eigen::Matrix<int, 1, 1>() << t).finished();
        model_parameters_[2] = (Eigen::Matrix<int, 1, 1>() << f).finished();
        model_parameters_[3] = mean;
        model_parameters_[4] = covariance;
    }

    double BinomialNormalPosterior::EvaluateModel(const Eigen::VectorXd &x) override
    {
        return ComputeBinomialPDF(x) * ComputeNormalPDF(x);
    }

    double BinomialNormalPosterior::EvaluateLogModel(const Eigen::VectorXd &x)
    {
        return std::log(ComputeBinomialPDF(x)) + std::log(ComputeNormalPDF(x)) ;
    }

    Eigen::VectorXd BinomialNormalPosterior::EvaluateModelGrad(const Eigen::VectorXd &x) override
    {
        // Not implementing because it's not required for the sensor degradation purpose
    }

protected:
    double ComputeBinomialPDF(const Eigen::VectorXd &x, const bool &normalized = false)
    {
        double p = x[0] * model_parameters_[2](0) + (1 - x[1]) * (1.0 - model_parameters_[2](0)); // b*f + (1-w) * (1-f)
        double binomial_pdf = std::pow(p, model_parameters_[0](0)) *
                              std::pow(1 - p, (model_parameters_[1] - model_parameters_[0])(0)); // p^n * (1 - p)^(t-n)

        return normalized ? binomial_pdf / ComputeBinomialNormalizationConstant() : binomial_pdf;
    }

    double ComputeNormalPDF(const Eigen::VectorXd &x, const bool &normalized = false)
    {
        Eigen::VectorXd diff = x - model_parameters_[3];
        double normal_pdf = ((-0.5 * (diff.transpose() * model_parameters_[4].inverse() * diff).array()).exp())(0); // exp(-0.5 * x^T * C^-1 * x)

        return normalized ? normal_pdf / ComputeNormalNormalizationConstant() : normal_pdf;
    }

    double ComputeNormalNormalizationConstant()
    {
        return 1.0 / (std::pow(2.0 * M_PI, dimension_ / 2.0) * std::sqrt(model_parameters_[1].determinant()));
    }

    double ComputeBinomialNormalizationConstant()
    {
    }
};

#endif
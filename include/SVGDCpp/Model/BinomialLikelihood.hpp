#ifndef SVGD_CPP_BINOMIAL_LIKELIHOOD_HPP
#define SVGD_CPP_BINOMIAL_LIKELIHOOD_HPP

#include "../Core.hpp"
#include "Model.hpp"

class BinomialLikelihood : public Model
{
public:

    BinomialLikelihood(const Eigen::Matrix<double, 1, 1> &n,
                       const Eigen::Matrix<double, 1, 1> &t,
                       const Eigen::Matrix<double, 1, 1> &f)
    {
        model_parameters_.push_back(n);
        model_parameters_.push_back(t);
        model_parameters_.push_back(f);
    }

    ~BinomialLikelihood() {}

    void Initialize() override
    {
        // do nothing because the base Initialize will call SetupADFun()
    }

    double EvaluateModel(const Eigen::VectorXd &x) override
    {
        return ComputePDF(x);
    }

    double EvaluateLogModel(const Eigen::VectorXd &x) override
    {
        return std::log(ComputePDF(x));
    }

    Eigen::VectorXd EvaluateModelGrad(const Eigen::VectorXd &x) override
    {
        // Not implementing because it's not required for the sensor degradation purpose
    }

    Eigen::VectorXd EvaluateLogModelGrad(const Eigen::VectorXd &x) override
    {
        return ComputeLogPDFGrad(x);
    }

protected:
    double ComputePDF(const Eigen::VectorXd &x)
    {
        double p = x(0) * model_parameters_[2](0) + (1 - x(1)) * (1.0 - model_parameters_[2](0)); // b*f + (1-w) * (1-f)

        return std::pow(p, model_parameters_[0](0)) *
               std::pow(1 - p, (model_parameters_[1] - model_parameters_[0])(0)); // p^n * (1 - p)^(t-n);
    }

    Eigen::Vector2d ComputeLogPDFGrad(const Eigen::Vector2d &x)
    {
        Eigen::Vector2d result;

        // Assuming b as the first parameter and w as the second parameter
        double n = model_parameters_[0](0);
        double t = model_parameters_[1](0);
        double f = model_parameters_[2](0);

        double q = f * (x(0) + x(1) - 1);
        double denom = (q + 1 - x(1)) * (q - x(1));
        double num = t * (q + 1 - x(1)) - n;

        result(0) = f * num / denom;
        result(1) = (f - 1) * num / denom;

        return result;
    }
};

#endif
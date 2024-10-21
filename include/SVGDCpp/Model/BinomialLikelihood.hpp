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
        : Model(2)
    {
        model_parameters_.push_back(ConvertToCppAD(n));
        model_parameters_.push_back(ConvertToCppAD(t));
        model_parameters_.push_back(ConvertToCppAD(f));

        // Define model function
        auto model_fun = [this](const VectorXADd &x, const std::vector<MatrixXADd> &params)
        {
            CppAD::AD<double> n = params[0](0);
            CppAD::AD<double> t = params[1](0);
            CppAD::AD<double> f = params[2](0);
            CppAD::AD<double> p = x(0) * f + (1.0 - x(1)) * (1.0 - f); // b*f + (1-w) * (1-f)

            VectorXADd result(1);
            result << CppAD::pow(p, n) * CppAD::pow(1.0 - p, t - n); // p^n * (1 - p)^(t-n);
            return result;
        };

        UpdateModel(model_fun);
    }

    ~BinomialLikelihood() {}
};

#endif
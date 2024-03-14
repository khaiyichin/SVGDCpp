#ifndef SVGD_CPP_ADAM_HPP
#define SVGD_CPP_ADAM_HPP

#include "../Core.hpp"
#include "Optimizer.hpp"

class Adam : public Optimizer
{
public:
    Adam(const double &lr, const double &beta1, const double &beta2, const double &epsilon = 1.0e-8)
        : Optimizer(lr, epsilon), decay_rate_1_(beta1), decay_rate_2_(beta2)
    {
        if (beta1 > 1.0 || beta1 < 0.0 || beta2 > 1.0 || beta2 < 0.0)
        {
            throw std::runtime_error("Invalid value for decay parameter beta.");
        }
    }

    virtual ~Adam() {}

    virtual Eigen::MatrixXd Step(const Eigen::MatrixXd &grad_matrix)
    {
        if (initial_run_)
        {
            sum_of_sq_grad_ = Eigen::MatrixXd::Zero(grad_matrix.rows(), grad_matrix.cols()).array();
            sum_of_grad_ = Eigen::MatrixXd::Zero(grad_matrix.rows(), grad_matrix.cols()).array();

            initial_run_ = false;
        }

        sum_of_grad_ = decay_rate_1_ * sum_of_grad_ + (1 - decay_rate_1_) * grad_matrix.array();                // Momentum part
        sum_of_sq_grad_ = decay_rate_2_ * sum_of_sq_grad_ + (1 - decay_rate_2_) * grad_matrix.array().square(); // RMSProp part

        ++counter_; // step increment

        return (learning_rate_ * (stabilizer_ + (CorrectForBias(sum_of_sq_grad_, decay_rate_2_)).sqrt()).inverse() * CorrectForBias(sum_of_grad_, decay_rate_1_)).matrix();
    }

protected:
    Eigen::ArrayXd CorrectForBias(const Eigen::ArrayXXd arr, const double &decay)
    {
        return arr / (1.0 - std::pow(decay, counter_));
    }

    size_t counter_ = 0;

    double decay_rate_1_;

    double decay_rate_2_;

    Eigen::ArrayXXd sum_of_sq_grad_;

    Eigen::ArrayXXd sum_of_grad_;
};

#endif
#ifndef SVGD_CPP_RMSPROP_HPP
#define SVGD_CPP_RMSPROP_HPP

#include "../Core.hpp"
#include "Optimizer.hpp"

class RMSProp : public Optimizer
{
public:
    RMSProp(const double &lr, const double &beta, const double &epsilon = 1.0e-8) : Optimizer(lr, epsilon), decay_rate_(beta)
    {
        if (beta > 1.0 || beta < 0.0)
        {
            throw std::runtime_error("Invalid value for decay parameter beta.");
        }
    }

    virtual ~RMSProp() {}

    virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd &grad_matrix)
    {
        if (initial_run_)
        {
            sum_of_sq_grad_ = Eigen::MatrixXd::Zero(grad_matrix.rows(), grad_matrix.cols()).array();

            initial_run_ = false;
        }

        sum_of_sq_grad_ = decay_rate_ * sum_of_sq_grad_ + (1 - decay_rate_) * grad_matrix.array().square();

        return (learning_rate_ * (stabilizer_ + sum_of_sq_grad_.sqrt()).inverse() * grad_matrix.array()).matrix();
    }

protected:
    double decay_rate_;

    Eigen::ArrayXXd sum_of_sq_grad_;
};

#endif
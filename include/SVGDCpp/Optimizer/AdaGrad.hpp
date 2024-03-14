#ifndef SVGD_CPP_ADAGRAD_HPP
#define SVGD_CPP_ADAGRAD_HPP

#include "../Core.hpp"
#include "Optimizer.hpp"

class AdaGrad : public Optimizer
{
public:
    AdaGrad(const double &lr, const double &epsilon = 1.0e-8) : Optimizer(lr, epsilon) {}

    virtual ~AdaGrad() {}

    virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd &grad_matrix)
    {
        if (initial_run_)
        {
            sum_of_sq_grad_ = Eigen::MatrixXd::Zero(grad_matrix.rows(), grad_matrix.cols()).array();

            initial_run_ = false;
        }

        sum_of_sq_grad_ = sum_of_sq_grad_ + grad_matrix.array().square();

        return (learning_rate_ * (stabilizer_ + sum_of_sq_grad_.sqrt()).inverse() * grad_matrix.array()).matrix();
    }

protected:
    double decay_rate_;

    Eigen::ArrayXXd sum_of_sq_grad_;
};

#endif
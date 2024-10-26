#ifndef SVGDCPP_ADAGRAD_HPP
#define SVGDCPP_ADAGRAD_HPP

#include "../Core.hpp"
#include "Optimizer.hpp"

class AdaGrad : public Optimizer
{
public:
    AdaGrad(const size_t &dimension,
            const size_t &num_particles,
            const double &lr,
            const double &epsilon = 1.0e-8)
        : Optimizer(lr, epsilon),
          dimension_(dimension),
          num_particles_(num_particles) {}

    virtual ~AdaGrad() {}

    virtual void Initialize() override
    {
        sum_of_sq_grad_ = Eigen::MatrixXd::Zero(dimension_, num_particles_).array();
    }

    virtual Eigen::MatrixXd Step(const Eigen::MatrixXd &grad_matrix)
    {
        sum_of_sq_grad_ += grad_matrix.array().square();

        return (learning_rate_ * (stabilizer_ + sum_of_sq_grad_.sqrt()).inverse() * grad_matrix.array()).matrix();
    }

protected:
    size_t dimension_;

    size_t num_particles_;

    double decay_rate_;

    Eigen::ArrayXXd sum_of_sq_grad_;
};

#endif
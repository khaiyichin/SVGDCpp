#ifndef SVGD_CPP_RMSPROP_HPP
#define SVGD_CPP_RMSPROP_HPP

#include "../Core.hpp"
#include "Optimizer.hpp"

class RMSProp : public Optimizer
{
public:
    RMSProp(const size_t &dimension,
            const size_t &num_particles,
            const double &lr,
            const double &beta,
            const double &epsilon = 1.0e-8)
        : Optimizer(lr, epsilon),
          dimension_(dimension),
          num_particles_(num_particles),
          decay_rate_(beta)
    {
        if (beta > 1.0 || beta < 0.0)
        {
            throw std::invalid_argument("SVGDCpp: [Argument Error] Invalid value for decay parameter beta.");
        }
    }

    virtual ~RMSProp() {}

    void Initialize() override
    {
        sum_of_sq_grad_ = Eigen::MatrixXd::Zero(dimension_, num_particles_).array();
    }

    virtual Eigen::MatrixXd Step(const Eigen::MatrixXd &grad_matrix)
    {
        sum_of_sq_grad_ = decay_rate_ * sum_of_sq_grad_ + (1 - decay_rate_) * grad_matrix.array().square();

        return (learning_rate_ * (stabilizer_ + sum_of_sq_grad_.sqrt()).inverse() * grad_matrix.array()).matrix();
    }

protected:
    size_t dimension_;

    size_t num_particles_;

    double decay_rate_;

    Eigen::ArrayXXd sum_of_sq_grad_;
};

#endif
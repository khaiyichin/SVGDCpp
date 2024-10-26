#ifndef SVGDCPP_OPTIMIZER_HPP
#define SVGDCPP_OPTIMIZER_HPP

#include "../Core.hpp"

class Optimizer
{
public:
    Optimizer(const double &lr, const double &epsilon = 1.0e-8) : learning_rate_(lr), stabilizer_(epsilon) {}

    virtual ~Optimizer() {}

    virtual void Initialize() = 0;

    virtual Eigen::MatrixXd Step(const Eigen::MatrixXd &grad_matrix) = 0;

protected:

    double learning_rate_;

    double stabilizer_;
};

#endif
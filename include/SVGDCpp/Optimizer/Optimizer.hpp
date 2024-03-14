#ifndef SVGD_CPP_OPTIMIZER_HPP
#define SVGD_CPP_OPTIMIZER_HPP

#include "../Core.hpp"

class Optimizer
{
public:
    Optimizer(const double &lr, const double &epsilon = 1.0e-8) : learning_rate_(lr), stabilizer_(epsilon) {}

    virtual ~Optimizer() {}

    virtual Eigen::MatrixXd Step(const Eigen::MatrixXd &grad_matrix) = 0;

protected:
    bool initial_run_ = true;

    double learning_rate_;

    double stabilizer_;
};

#endif
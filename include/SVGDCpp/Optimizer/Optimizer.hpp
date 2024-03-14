#ifndef SVGD_CPP_OPTIMIZER_HPP
#define SVGD_CPP_OPTIMIZER_HPP

#include "../Core.hpp"

class Optimizer
{
public:
    Optimizer(const double &lr, const double &epsilon = 1.0e-8) : learning_rate_(lr), stabilizer_(epsilon) {}

    virtual ~Optimizer() {}

    virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd &grad_matrix)
    {
        throw std::runtime_error("Optimizer functor `operator()` method must be provided in derived class.");
    }

protected:
    bool initial_run_ = true;

    double learning_rate_;

    double stabilizer_;
};

#endif
/**
 * @file Optimizer.hpp
 * @author Khai Yi Chin (khaiyichin@gmail.com)
 * @brief Optimizer class header
 *
 * @copyright Copyright (c) 2024 Khai Yi Chin
 *
 */
#ifndef SVGDCPP_OPTIMIZER_HPP
#define SVGDCPP_OPTIMIZER_HPP

#include "../Core.hpp"

/**
 * @class Optimizer
 * @brief This is used as a template to derive optimizer classes. Specifically, the `Step` function needs to be defined.
 * @ingroup Optimizer_Module
 */
class Optimizer
{
public:
    /**
     * @brief Default constructor.
     *
     * @param lr Learning rate.
     * @param epsilon Numerical stabilizer.
     */
    Optimizer(const double &lr, const double &epsilon = 1.0e-8) : learning_rate_(lr), stabilizer_(epsilon) {}

    /**
     * @brief Default destructor.
     *
     */
    virtual ~Optimizer() {}

    /**
     * @brief Initialize the optimizer.
     *
     */
    virtual void Initialize() = 0;

    virtual Eigen::MatrixXd Step(const Eigen::MatrixXd &grad_matrix) = 0;

protected:
    double learning_rate_; ///< Learning rate.

    double stabilizer_; /// Numerical stabilizer.
};

#endif
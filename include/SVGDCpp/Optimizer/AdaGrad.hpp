/**
 * @file AdaGrad.hpp
 * @author Khai Yi Chin (khaiyichin@gmail.com)
 * @brief AdaGrad optimizer class header.
 *
 * @copyright Copyright (c) 2024 Khai Yi Chin
 *
 */
#ifndef SVGDCPP_ADAGRAD_HPP
#define SVGDCPP_ADAGRAD_HPP

#include "../Core.hpp"
#include "Optimizer.hpp"

/**
 * @class AdaGrad
 * @brief This class provides the AdaGrad optimizer.
 * @ingroup Optimizer_Module
 */
class AdaGrad : public Optimizer
{
public:
    /**
     * @brief Default constructor.
     *
     * @param dimension Dimension of the problem.
     * @param num_particles Number of particles used in the problem.
     * @param lr Learning rate.
     * @param epsilon Numerical stabilizer.
     */
    AdaGrad(const size_t &dimension,
            const size_t &num_particles,
            const double &lr,
            const double &epsilon = 1.0e-8)
        : Optimizer(lr, epsilon),
          dimension_(dimension),
          num_particles_(num_particles) {}

    /**
     * @brief Default destructor.
     *
     */
    virtual ~AdaGrad() {}

    /**
     * @brief Initialize the optimizer.
     *
     */
    virtual void Initialize() override
    {
        sum_of_sq_grad_ = Eigen::MatrixXd::Zero(dimension_, num_particles_).array();
    }

    /**
     * @brief Execute the optimization of the gradient at each step.
     *
     * @param grad_matrix Gradient matrix for each particle coordinate.
     * @return Optimized gradient matrix.
     */
    virtual Eigen::MatrixXd Step(const Eigen::MatrixXd &grad_matrix)
    {
        sum_of_sq_grad_ += grad_matrix.array().square();

        return (learning_rate_ * (stabilizer_ + sum_of_sq_grad_.sqrt()).inverse() * grad_matrix.array()).matrix();
    }

protected:
    size_t dimension_; ///< Dimension of the problem.

    size_t num_particles_; ///< Number of particles used in the problem.

    double decay_rate_; ///< Decay rate (unused).

    Eigen::ArrayXXd sum_of_sq_grad_; ///< Sum of squares gradient matrix.
};

#endif
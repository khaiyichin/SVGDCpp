/**
 * @file Adam.hpp
 * @author Khai Yi Chin (khaiyichin@gmail.com)
 * @brief Adam optimizer class header.
 *
 * @copyright Copyright (c) 2024 Khai Yi Chin
 *
 */
#ifndef SVGDCPP_ADAM_HPP
#define SVGDCPP_ADAM_HPP

#include "../Core.hpp"
#include "Optimizer.hpp"

/**
 * @class Adam
 * @brief This class provides the Adam optimizer.
 * @ingroup Optimizer_Module
 */
class Adam : public Optimizer
{
public:
    /**
     * @brief Default constructor.
     *
     * @param dimension Dimension of the problem.
     * @param num_particles Number of particles used in the problem.
     * @param lr Learning rate.
     * @param beta1 Exponential decay rate for the 1st moment estimates.
     * @param beta2 Exponential decay rate for the 2nd moment estimates.
     * @param epsilon Numerical stabilizer.
     */
    Adam(const size_t &dimension,
         const size_t &num_particles,
         const double &lr,
         const double &beta1,
         const double &beta2,
         const double &epsilon = 1.0e-8)
        : Optimizer(lr, epsilon),
          dimension_(dimension),
          num_particles_(num_particles),
          decay_rate_1_(beta1),
          decay_rate_2_(beta2)
    {
        if (beta1 >= 1.0 || beta1 < 0.0 || beta2 >= 1.0 || beta2 < 0.0)
        {
            throw std::invalid_argument(SVGDCPP_LOG_PREFIX + "[Argument Error] Invalid value for decay parameter beta.");
        }
    }

    /**
     * @brief Default destructor.
     *
     */
    virtual ~Adam() {}

    /**
     * @brief Initialize the optimizer.
     *
     */
    virtual void Initialize() override
    {
        sum_of_sq_grad_ = Eigen::MatrixXd::Zero(dimension_, num_particles_).array();
        sum_of_grad_ = Eigen::MatrixXd::Zero(dimension_, num_particles_).array();

        counter_ = 0;
    }

    /**
     * @brief Execute the optimization of the gradient at each step.
     *
     * @param grad_matrix Gradient matrix for each particle coordiniate.
     * @return Optimized gradient matrix.
     */
    virtual Eigen::MatrixXd Step(const Eigen::MatrixXd &grad_matrix)
    {
        sum_of_grad_ = decay_rate_1_ * sum_of_grad_ + (1 - decay_rate_1_) * grad_matrix.array();                // Momentum part
        sum_of_sq_grad_ = decay_rate_2_ * sum_of_sq_grad_ + (1 - decay_rate_2_) * grad_matrix.array().square(); // RMSProp part

        ++counter_; // step increment

        return (learning_rate_ * (stabilizer_ + (CorrectForBias(sum_of_sq_grad_, decay_rate_2_)).sqrt()).inverse() * CorrectForBias(sum_of_grad_, decay_rate_1_)).matrix();
    }

protected:
    /**
     * @brief Correct moment estimate for bias.
     *
     * @param arr Biased gradient estimate array.
     * @param decay Decay rate.
     * @return Bias-corrected moment estimate.
     */
    Eigen::ArrayXXd CorrectForBias(const Eigen::ArrayXXd arr, const double &decay)
    {
        return arr / (1.0 - std::pow(decay, counter_));
    }

    size_t counter_ = 0; ///< Time step counter.

    size_t dimension_; ///< Dimension of the problem.

    size_t num_particles_; ///< Number of particles used in the problem.

    double decay_rate_1_; ///< Exponential decay rate for the 1st moment estimate.

    double decay_rate_2_; ///< Exponential decay rate for the 2nd moment estimate.

    Eigen::ArrayXXd sum_of_sq_grad_; ///< 2nd biased moment estimate.

    Eigen::ArrayXXd sum_of_grad_; ///< 1st biased moment estimate
};

#endif
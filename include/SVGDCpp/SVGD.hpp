/**
 * @file SVGD.hpp
 * @author Khai Yi Chin (khaiyichin@gmail.com)
 * @brief SVGD class header
 * @version 0.1
 * @date 2024-03-22
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef SVGDCPP_SVGD_HPP
#define SVGDCPP_SVGD_HPP

#include "Core.hpp"
#include "Kernel/Kernel.hpp"
#include "Model/Model.hpp"
#include "Optimizer/Optimizer.hpp"

/**
 * @class SVGD
 * @brief Main class to run SVGD on particle coordinates.
 * @details To run SVGD successfully, 3 things are required: a @ref Kernel (or its derived class) object, a @ref Model (or its derived class) object, and an @ref Optimizer (or its derived class) object.
 * ```cpp
 * // Set up a 2-D multivariate normal problem with a RBF kernel using 10 particles and the Adam optimizer for 1000 iterations
 * size_t dim = 2, num_particles = 10, num_iterations = 1000;
 * auto x0 = std::make_shared<Eigen::MatrixXd>(3*Eigen::MatrixXd::Random(dim, num_particles));
 *
 * // Create multivariate normal model pointer
 * Eigen::Vector2d mean(5, -5);
 * Eigen::Matrix2d covariance;
 * covariance << 0.5, 0, 0, 0.5;
 * std::shared_ptr<Model> model_ptr = std::make_shared<MultivariateNormal>(mean, covariance);
 *
 * // Create RBF kernel pointer
 * std::shared_ptr<Kernel> kernel_ptr = std::make_shared<GaussianRBFKernel>(x0, GaussianRBFKernel::ScaleMethod::Median, model_ptr);
 *
 * // Create Adam optimizer pointer
 * std::shared_ptr<Optimizer>opt_ptr = std::make_shared<Adam>(dim, num_particles, 1.0e-1, 0.9, 0.999);
 *
 * // Instantiate the SVGD class
 * SVGD svgd(dim, num_iterations, x0, kernel_ptr, model_ptr, opt_ptr);
 *
 * // Initialize and run SVGD for 1000 iterations
 * svgd.Initialize();
 * svgd.Run();
 * ```
 * @ingroup Core_Module
 */
class SVGD
{
public:
    /**
     * @brief Construct a new SVGD object.
     * @details Overloads the main constructor so that users can create this object without specifying bounds.
     *
     * @param dim Dimension of particle coordinates; should match the number of rows in the coordinate matrix.
     * @param iter Number of iterations to run SVGD.
     * @param coord_mat_ptr Pointer to the particle coordinate matrix.
     * @param kernel_ptr Pointer to a @ref Kernel (or its derived class) object.
     * @param model_ptr Pointer to a @ref Model (or its derived class) object.
     * @param optimizer_ptr Pointer to an @ref Optimizer (or its derived class) object.
     * @param parallel Flag to run SVGD in multi-threaded mode.
     */
    SVGD(
        const size_t &dim,
        const size_t &iter,
        const std::shared_ptr<Eigen::MatrixXd> &coord_mat_ptr,
        const std::shared_ptr<Kernel> &kernel_ptr,
        const std::shared_ptr<Model> &model_ptr,
        const std::shared_ptr<Optimizer> &optimizer_ptr,
        const bool &parallel = false)
        : SVGD(dim, iter, coord_mat_ptr, kernel_ptr, model_ptr, optimizer_ptr, Eigen::Matrix<double, 1, 1>(-INFINITY), Eigen::Matrix<double, 1, 1>(INFINITY), parallel) {}

    /**
     * @brief Construct a new SVGD object.
     *
     * @param dim Dimension of particle coordinates; should match the number of rows in the coordinate matrix.
     * @param iter Number of iterations to run SVGD.
     * @param coord_mat_ptr Pointer to the particle coordinate matrix.
     * @param kernel_ptr Pointer to a @ref Kernel (or its derived class) object.
     * @param model_ptr Pointer to a @ref Model (or its derived class) object.
     * @param optimizer_ptr Pointer to an @ref Optimizer (or its derived class) object.
     * @param bound_lower Lower bound for the problem.
     * @param bound_upper Upper bound for the problem.
     * @param parallel Flag to run SVGD in multi-threaded mode.
     */
    SVGD(
        const size_t &dim,
        const size_t &iter,
        const std::shared_ptr<Eigen::MatrixXd> &coord_mat_ptr,
        const std::shared_ptr<Kernel> &kernel_ptr,
        const std::shared_ptr<Model> &model_ptr,
        const std::shared_ptr<Optimizer> &optimizer_ptr,
        const Eigen::VectorXd &bound_lower,
        const Eigen::VectorXd &bound_upper,
        const bool &parallel = false)
        : dimension_(coord_mat_ptr->rows()),
          num_iterations_(iter),
          parallel_(parallel)
    {
        if (dimension_ != dim)
        {
            throw DimensionMismatchException("Specified dimension does not match the particle coordinate matrix.");
        }

        // Assign bounds
        if (bound_lower.rows() == 1 &&
            bound_lower == Eigen::Matrix<double, 1, 1>(-INFINITY) &&
            bound_upper.rows() == 1 &&
            bound_upper == Eigen::Matrix<double, 1, 1>(INFINITY))
        {
            check_bounds_ = false; // avoid unnecessary bound checking if bounds are default
        }
        else
        {
            if (bound_lower.rows() != dimension_ && bound_lower.rows() != 1)
            {
                throw DimensionMismatchException("The provided lower bounds have incorrect dimensions.");
            }
            else
            {
                std::cout << SVGDCPP_LOG_PREFIX + "Lower bound set to " << bound_lower.transpose() << std::endl;
                check_bounds_ = true;
            }

            bounds_.first = bound_lower;

            if (bound_upper.rows() != dimension_ && bound_upper.rows() != 1)
            {
                throw DimensionMismatchException("The provided upper bounds have incorrect dimensions.");
            }
            else
            {
                std::cout << SVGDCPP_LOG_PREFIX + "Upper bound set to " << bound_upper.transpose() << std::endl;
                check_bounds_ = true;
            }

            bounds_.second = bound_upper;
        }

        coord_matrix_ptr_ = coord_mat_ptr; // coordinate matrix of n particles in a m-dimensional problem is m x n

        log_model_grad_matrix_.resize(dimension_, coord_matrix_ptr_->cols());                                             // m x n
        kernel_matrix_.resize(coord_matrix_ptr_->cols(), coord_matrix_ptr_->cols());                                      // n x n
        kernel_grad_matrix_.resize(dimension_ * coord_matrix_ptr_->cols(), coord_matrix_ptr_->cols());                    // (m*n) x n
        kernel_grad_indexer_ = Eigen::MatrixXd::Identity(dimension_, dimension_).replicate(1, coord_matrix_ptr_->cols()); // m x (m*n)

        // Store the kernel, distribution, and optimizer objects
        kernel_ptr_ = kernel_ptr;
        model_ptr_ = model_ptr;
        optimizer_ptr_ = optimizer_ptr;

        // Setup CppAD for parallel usage
        if (parallel_)
        {
            std::cout << SVGDCPP_LOG_PREFIX << Eigen::nbThreads() << " threads available for use." << std::endl;
            // Parallel application inspired from https://github.com/coin-or/CppAD/issues/197#issuecomment-1983462984

            // Create copies of the original kernel for n particles
            for (size_t i = 0; i < coord_matrix_ptr_->cols(); ++i)
            {
                model_ptr_vector_.push_back(model_ptr_->CloneUniquePointer());
                kernel_ptr_vector_.push_back(kernel_ptr_->CloneUniquePointer());
            }
        }
    }

    /**
     * @brief Default constructor.
     *
     */
    ~SVGD() {}

    /**
     * @brief Initializes the @ref Model and @ref Kernel objects.
     *
     */
    void Initialize()
    {
        // Initialize the optimizer
        optimizer_ptr_->Initialize();

        if (parallel_)
        {
#pragma omp parallel for
            for (int i = 0; i < coord_matrix_ptr_->cols(); ++i)
            {
                // Initialize the model
                model_ptr_vector_[i]->Initialize();

                // Initialize the kernel
                kernel_ptr_vector_[i]->Initialize();
            }
        }
        else
        {
            // Initialize the model
            model_ptr_->Initialize();

            // Initialize the kernel
            kernel_ptr_->Initialize();
        }
    }

    /**
     * @brief Update the kernel parameters.
     * @details Users can call this to change the parameters of the kernel.
     *
     * @param params Vector of variable-sized Eigen matrices.
     */
    void UpdateKernelParameters(const std::vector<Eigen::MatrixXd> &params)
    {
        if (parallel_)
        {
#pragma omp parallel for
            for (int i = 0; i < coord_matrix_ptr_->cols(); ++i)
            {
                kernel_ptr_vector_[i]->UpdateParameters(params);
                kernel_ptr_vector_[i]->Initialize();
            }
        }
        else
        {
            kernel_ptr_->UpdateParameters(params);
            kernel_ptr_->Initialize();
        }
    }

    /**
     * @brief Update the model parameters.
     * @details Users can call this to change the parameters of the model.
     *
     * @param params Vector of variable-sized Eigen matrices.
     */
    void UpdateModelParameters(const std::vector<Eigen::MatrixXd> &params)
    {
        if (parallel_)
        {
#pragma omp parallel for
            for (int i = 0; i < coord_matrix_ptr_->cols(); ++i)
            {
                model_ptr_vector_[i]->UpdateParameters(params);
                model_ptr_vector_[i]->Initialize();
            }
        }
        else
        {
            model_ptr_->UpdateParameters(params);
            model_ptr_->Initialize();
        }
    }

    /**
     * @brief Execute SVGD on the particle locations.
     *
     */
    void Run()
    {
        // Run L iterations
        for (size_t iter = 0; iter < num_iterations_; ++iter)
        {
            Step(); // x + e * phi(x)
        }
    }

protected:
    /**
     * @brief Execute a SVGD step.
     *
     */
    void Step()
    {
        if (parallel_)
        {
#pragma omp parallel for
            for (size_t i = 0; i < coord_matrix_ptr_->cols(); ++i)
            {
                model_ptr_vector_[i]->Step();
                kernel_ptr_vector_[i]->Step();
            }
        }
        else
        {
            // Run model step functions
            model_ptr_->Step();

            // Run kernel step functions
            kernel_ptr_->Step();
        }

        // Update particle positions
        *coord_matrix_ptr_ += optimizer_ptr_->Step(ComputePhi());

        // Check bounds
        if (check_bounds_)
        {
            for (int i = 0; i < coord_matrix_ptr_->rows(); ++i)
            {
                coord_matrix_ptr_->row(i) = (coord_matrix_ptr_->row(i).array() < bounds_.first(i)).select(bounds_.first(i), coord_matrix_ptr_->row(i));
                coord_matrix_ptr_->row(i) = (coord_matrix_ptr_->row(i).array() > bounds_.second(i)).select(bounds_.second(i), coord_matrix_ptr_->row(i));
            }
        }
    }

    /**
     * @brief Compute the Stein variational gradient for one iteration.
     *
     * @return Matrix of gradients with the size of (dimension) x (num of particles).
     */
    Eigen::MatrixXd ComputePhi()
    {
        // Go through each particle
        if (parallel_)
        {
#pragma omp parallel for
            for (int i = 0; i < coord_matrix_ptr_->cols(); ++i)
            {
                // Compute log pdf grad
                log_model_grad_matrix_.block(0, i, dimension_, 1) = model_ptr_vector_[i]->EvaluateLogModelGrad(coord_matrix_ptr_->col(i));

                kernel_ptr_vector_[i]->UpdateLocation(coord_matrix_ptr_->col(i));
                kernel_ptr_vector_[i]->Initialize();

                // Iterate through each neighbor of the particle (including itself) to compute kernel related values
                for (int j = 0; j < coord_matrix_ptr_->cols(); ++j)
                {
                    kernel_matrix_(j, i) = kernel_ptr_vector_[i]->EvaluateKernel(coord_matrix_ptr_->col(j));                                            // k(x_j, x_i)
                    kernel_grad_matrix_.block(j * dimension_, i, dimension_, 1) = kernel_ptr_vector_[i]->EvaluateKernelGrad(coord_matrix_ptr_->col(j)); // grad k(x_j, x_i)
                }
            }
        }
        else
        {
            for (int i = 0; i < coord_matrix_ptr_->cols(); ++i)
            {
                // Compute log pdf grad
                log_model_grad_matrix_.block(0, i, dimension_, 1) = model_ptr_->EvaluateLogModelGrad(coord_matrix_ptr_->col(i));

                kernel_ptr_->UpdateLocation(coord_matrix_ptr_->col(i));
                kernel_ptr_->Initialize();

                // Iterate through each neighbor of the particle (including itself) to compute kernel related values
                for (int j = 0; j < coord_matrix_ptr_->cols(); ++j)
                {
                    kernel_matrix_(j, i) = kernel_ptr_->EvaluateKernel(coord_matrix_ptr_->col(j));                                            // k(x_j, x_i)
                    kernel_grad_matrix_.block(j * dimension_, i, dimension_, 1) = kernel_ptr_->EvaluateKernelGrad(coord_matrix_ptr_->col(j)); // grad k(x_j, x_i)
                }
            }
        }

        // std::cout << "debug log_model_grad_matrix_\n" << log_model_grad_matrix_ << "\nkernel_matrix_\n" << kernel_matrix_ << "\nkernel_grad_matrix_\n" << kernel_grad_matrix_ << std::endl;

        return (1.0 / coord_matrix_ptr_->cols()) * (log_model_grad_matrix_ * kernel_matrix_ + kernel_grad_indexer_ * kernel_grad_matrix_);
    }

    int dimension_ = -1; ///< Dimension of the particle coordinates.

    size_t num_iterations_; ///< Number of iterations to run SVGD.

    const bool parallel_ = false; ///< Flag to indicate whether to run SVGD with multiple threads.

    bool check_bounds_ = false; ///< Flag to indicate whether bounds checking is necessary.

    // Idea: we store N kernel function objects for N corresponding particles; the particles' state is used to parametrize
    // the kernels. Here we're trying to trade memory for speed; with a kernel 'responsible' for each particle we don't
    // need to keep updating the kernel parameters
    std::shared_ptr<Kernel> kernel_ptr_; ///< Pointer to the kernel object.

    std::shared_ptr<Model> model_ptr_; ///< Pointer to the Model object.

    std::shared_ptr<Optimizer> optimizer_ptr_; ///< Pointer to the Optimizer object.

    std::shared_ptr<Eigen::MatrixXd> coord_matrix_ptr_; ///< Pointer to the particle coordinate matrix; shape is @a m (@ref dimension_) x @a n (number of particles).

    std::pair<Eigen::VectorXd, Eigen::VectorXd> bounds_; ///< Pair of bounds (lower, upper) to the problem.

    Eigen::MatrixXd log_model_grad_matrix_; ///< Matrix containing the gradients of the log model function; shape is @a m x @a n.

    Eigen::MatrixXd kernel_matrix_; ///< Matrix containing the values of the kernel function; shape is @a n x @a n.

    Eigen::MatrixXd kernel_grad_matrix_; ///< Matrix containing the gradients of the kernel function; shape is (@a m x @a n) x @a n.

    Eigen::MatrixXd kernel_grad_indexer_; ///< Matrix to index the gradients of the kernel function; shape is @a m x (@a m x @a n).

    std::vector<std::unique_ptr<Kernel>> kernel_ptr_vector_; ///< Vector of pointers to Kernel instance copies for parallel computation

    std::vector<std::unique_ptr<Model>> model_ptr_vector_; ///< Vector of pointers to Model instance copies for parallel computation
};

#endif
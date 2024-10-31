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

#include <fstream>
#include <sstream>

#include "Core.hpp"
#include "Kernel/Kernel.hpp"
#include "Model/Model.hpp"
#include "Optimizer/Optimizer.hpp"

/**
 * @struct SVGDOptions
 * @brief Struct to provide options to run SVGD.
 *
 * @ingroup Core_Module
 */
struct SVGDOptions
{
    size_t Dimension;

    size_t NumIterations;

    std::shared_ptr<Eigen::MatrixXd> CoordinateMatrixPtr = nullptr;

    std::shared_ptr<Kernel> KernelPtr = nullptr;

    std::shared_ptr<Model> ModelPtr = nullptr;

    std::shared_ptr<Optimizer> OptimizerPtr = nullptr;

    Eigen::VectorXd LowerBound = Eigen::VectorXd::Constant(1, -INFINITY);

    Eigen::VectorXd UpperBound = Eigen::VectorXd::Constant(1, INFINITY);

    std::string IntermediateMatricesOutputPath = "log.txt";

    bool Parallel = false;

    bool LogIntermediateMatrices = false;

    SVGDOptions() {}
};

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
     * @details Overloads the main constructor so that users can create this object with the @ref SVGDOptions struct.
     *
     * @param options_struct @ref SVGDOptions struct.
     */
    SVGD(const SVGDOptions &options_struct)
        : SVGD(options_struct.Dimension,
               options_struct.NumIterations,
               options_struct.CoordinateMatrixPtr,
               options_struct.KernelPtr,
               options_struct.ModelPtr,
               options_struct.OptimizerPtr,
               options_struct.LowerBound,
               options_struct.UpperBound,
               options_struct.Parallel,
               options_struct.LogIntermediateMatrices,
               options_struct.IntermediateMatricesOutputPath) {}

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
     * @param parallel Flag to run SVGD in multithreaded mode.
     */
    SVGD(
        const size_t &dim,
        const size_t &iter,
        const std::shared_ptr<Eigen::MatrixXd> &coord_mat_ptr,
        const std::shared_ptr<Kernel> &kernel_ptr,
        const std::shared_ptr<Model> &model_ptr,
        const std::shared_ptr<Optimizer> &optimizer_ptr,
        const bool &parallel = false)
        : SVGD(dim,
               iter,
               coord_mat_ptr,
               kernel_ptr,
               model_ptr,
               optimizer_ptr,
               Eigen::VectorXd::Constant(1, -INFINITY),
               Eigen::VectorXd::Constant(1, INFINITY),
               parallel) {}

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
     * @param parallel Flag to run SVGD in multithreaded mode.
     * @param log_intermediate_matrices Flag to log intermediate computation results.
     * @param intermediate_matrices_output_path File to write intermediate computation results to.
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
        const bool &parallel = false,
        const bool &log_intermediate_matrices = false,
        const std::string &intermediate_matrices_output_path = "log.txt")
        : dimension_(coord_mat_ptr->rows()),
          num_iterations_(iter),
          parallel_(parallel),
          log_intermediate_matrices_(log_intermediate_matrices),
          intermediate_matrices_output_path_(intermediate_matrices_output_path)
    {
        // Check dimensions
        if (dimension_ != dim)
        {
            throw DimensionMismatchException("Specified dimension does not match the particle coordinate matrix.");
        }

        // Initialize matrices
        coord_matrix_ptr_ = coord_mat_ptr; // coordinate matrix of n particles in a m-dimensional problem is m x n

        log_model_grad_matrix_.resize(dimension_, coord_matrix_ptr_->cols());                                             // m x n
        kernel_matrix_.resize(coord_matrix_ptr_->cols(), coord_matrix_ptr_->cols());                                      // n x n
        kernel_grad_matrix_.resize(dimension_ * coord_matrix_ptr_->cols(), coord_matrix_ptr_->cols());                    // (m*n) x n
        kernel_grad_indexer_ = Eigen::MatrixXd::Identity(dimension_, dimension_).replicate(1, coord_matrix_ptr_->cols()); // m x (m*n)

        // Assign bounds
        if (bound_lower.rows() == 1 &&
            bound_lower == Eigen::VectorXd::Constant(1, -INFINITY) &&
            bound_upper.rows() == 1 &&
            bound_upper == Eigen::VectorXd::Constant(1, INFINITY))
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
                std::cout << SVGDCPP_LOG_PREFIX + "Bound checking enabled, lower bound set to " << bound_lower.transpose() << "." << std::endl;
                check_bounds_ = true;
            }

            bounds_.first = bound_lower.replicate(1, coord_matrix_ptr_->cols());

            if (bound_upper.rows() != dimension_ && bound_upper.rows() != 1)
            {
                throw DimensionMismatchException("The provided upper bounds have incorrect dimensions.");
            }
            else
            {
                std::cout << SVGDCPP_LOG_PREFIX + "Bound checking enabled, upper bound set to " << bound_upper.transpose() << "." << std::endl;
                check_bounds_ = true;
            }

            bounds_.second = bound_upper.replicate(1, coord_matrix_ptr_->cols());
        }

        // Store the kernel, distribution, and optimizer objects
        kernel_ptr_ = kernel_ptr;
        model_ptr_ = model_ptr;
        optimizer_ptr_ = optimizer_ptr;

        if (kernel_ptr_ == nullptr)
        {
            throw std::invalid_argument(SVGDCPP_LOG_PREFIX + "[Argument Error] Invalid Kernel object pointer.");
        }

        if (model_ptr_ == nullptr)
        {
            throw std::invalid_argument(SVGDCPP_LOG_PREFIX + "[Argument Error] Invalid Model object pointer.");
        }

        if (optimizer_ptr_ == nullptr)
        {
            throw std::invalid_argument(SVGDCPP_LOG_PREFIX + "[Argument Error] Invalid Optimizer object pointer.");
        }

        // Setup CppAD for parallel usage
        if (parallel_)
        {
            std::cout << SVGDCPP_LOG_PREFIX << Eigen::nbThreads() << " threads available for use." << std::endl;
            // Parallel application inspired from https://github.com/coin-or/CppAD/issues/197#issuecomment-1983462984

            // Create copies of the original kernel for n particles
            for (size_t i = 0; i < coord_matrix_ptr_->cols(); ++i)
            {
                kernel_ptr_vector_.push_back(kernel_ptr_->CloneUniquePointer());
            }
        }
    }

    /**
     * @brief Prohibit copying of SVGD object to prevent unintended sharing of member variables.
     *
     */
    SVGD(const SVGD &) = delete;

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
        // Initialize the model
        model_ptr_->Initialize();

        // Initialize the kernel
        if (parallel_)
        {
#pragma omp parallel for
            for (int i = 0; i < coord_matrix_ptr_->cols(); ++i)
            {
                kernel_ptr_vector_[i]->Initialize();
            }
        }
        else
        {
            kernel_ptr_->Initialize();
        }

        // Initialize the optimizer
        optimizer_ptr_->Initialize();

        // Initialize container for logged intermediate matrices
        if (log_intermediate_matrices_)
        {
            intermediate_matrices_sstream_vector_.clear();
            intermediate_matrices_sstream_vector_.resize(num_iterations_);
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
        model_ptr_->UpdateParameters(params);
        model_ptr_->Initialize();
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

            // Log intermediate computation results
            if (log_intermediate_matrices_)
            {
                intermediate_matrices_sstream_vector_[iter] << "========== Step " << iter + 1 << " =========="
                                                            << "\nLogModelGrad=\n"
                                                            << log_model_grad_matrix_
                                                            << "\n\nKernel=\n"
                                                            << kernel_matrix_
                                                            << "\n\nKernelGrad=\n"
                                                            << kernel_grad_matrix_
                                                            << "\n\nCoordMat=\n"
                                                            << *coord_matrix_ptr_
                                                            << "\n\n";
            }
        }

        // Write logged intermediate computation results to file
        if (log_intermediate_matrices_)
        {
            WriteIntermediateMatricesToFile();
        }
    }

protected:
    /**
     * @brief Execute a SVGD step.
     *
     */
    void Step()
    {
        // Run model step functions
        model_ptr_->Step();

        if (parallel_)
        {
#pragma omp parallel for
            for (size_t i = 0; i < coord_matrix_ptr_->cols(); ++i)
            {
                kernel_ptr_vector_[i]->Step();
            }
        }
        else
        {
            // Run kernel step functions
            kernel_ptr_->Step();
        }

        // Update particle positions
        *coord_matrix_ptr_ += optimizer_ptr_->Step(ComputePhi());

        // Check bounds
        if (check_bounds_)
        {
            *coord_matrix_ptr_ = (coord_matrix_ptr_->array().min(bounds_.second.array()).max(bounds_.first.array())).matrix();
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
            // Compute log pdf grad in sequential mode because there cannot be copies made of model (in case kernel objects reference the model)
            for (int i = 0; i < coord_matrix_ptr_->cols(); ++i)
            {
                log_model_grad_matrix_.block(0, i, dimension_, 1) = model_ptr_->EvaluateLogModelGrad(coord_matrix_ptr_->col(i));
            }

#pragma omp parallel for
            for (int i = 0; i < coord_matrix_ptr_->cols(); ++i)
            {
                // Update each kernel's location vector
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

                // Update the kernel location vector
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

        return (1.0 / coord_matrix_ptr_->cols()) * (log_model_grad_matrix_ * kernel_matrix_ + kernel_grad_indexer_ * kernel_grad_matrix_);
    }

    /**
     * @brief Write the logged intermediate computation results.
     *
     */
    void WriteIntermediateMatricesToFile()
    {
        std::ofstream output_file(intermediate_matrices_output_path_);

        if (!output_file)
        {
            throw std::runtime_error(SVGDCPP_LOG_PREFIX + "[Runtime Error] Cannot open " + intermediate_matrices_output_path_ + " for writing.");
        }

        // Write to file
        for (const auto &ss : intermediate_matrices_sstream_vector_)
        {
            output_file << ss.str();
        }

        output_file.close();
    }

    int dimension_ = -1; ///< Dimension of the particle coordinates.

    size_t num_iterations_; ///< Number of iterations to run SVGD.

    const bool parallel_ = false; ///< Flag to indicate whether to run SVGD with multiple threads.

    bool check_bounds_ = false; ///< Flag to indicate whether bounds checking is necessary.

    bool log_intermediate_matrices_ = false; ///< Flag to indicate whether to log intermediate computation results. Useful for debugging.

    std::shared_ptr<Kernel> kernel_ptr_; ///< Pointer to the kernel object.

    std::shared_ptr<Model> model_ptr_; ///< Pointer to the Model object.

    std::shared_ptr<Optimizer> optimizer_ptr_; ///< Pointer to the Optimizer object.

    std::shared_ptr<Eigen::MatrixXd> coord_matrix_ptr_; ///< Pointer to the particle coordinate matrix; shape is @a m (@ref dimension_) x @a n (number of particles).

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> bounds_; ///< Pair of bounds (lower, upper) to the problem.

    Eigen::MatrixXd log_model_grad_matrix_; ///< Matrix containing the gradients of the log model function; shape is @a m x @a n.

    Eigen::MatrixXd kernel_matrix_; ///< Matrix containing the values of the kernel function; shape is @a n x @a n.

    Eigen::MatrixXd kernel_grad_matrix_; ///< Matrix containing the gradients of the kernel function; shape is (@a m x @a n) x @a n.

    Eigen::MatrixXd kernel_grad_indexer_; ///< Matrix to index the gradients of the kernel function; shape is @a m x (@a m x @a n).

    std::vector<std::unique_ptr<Kernel>> kernel_ptr_vector_; ///< Vector of pointers to Kernel instance copies for parallel computation.

    std::vector<std::stringstream> intermediate_matrices_sstream_vector_; ///< Vector of stringstreams containing intermediate computation results at each step.

    std::string intermediate_matrices_output_path_ = "log.txt"; ///< Output path for logged intermediate computation results.
};

#endif
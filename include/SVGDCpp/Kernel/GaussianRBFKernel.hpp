/**
 * @file GaussianRBFKernel.hpp
 * @author Khai Yi Chin (khaiyichin@gmail.com)
 * @brief Gaussian RBF kernel class header.
 * @version 0.1
 * @date 2024-03-23
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef SVGDCPP_GAUSSIAN_RBF_KERNEL_HPP
#define SVGDCPP_GAUSSIAN_RBF_KERNEL_HPP

#include "../Core.hpp"
#include "../Model/Model.hpp"
#include "Kernel.hpp"

/**
 * @class GaussianRBFKernel
 * @brief Implementation of a Gaussian RBF kernel.
 * @ingroup Kernel_Module
 */
class GaussianRBFKernel : public Kernel
{
public:
    enum class ScaleMethod
    {
        Median = 0, ///< @enum Median Median heuristic-based scale computation.
        Hessian = 1 ///< @enum Hessian Hessian-based scale computation.
        // TODO: constant scale
    };

    /**
     * @brief Default constructor.
     *
     * @details This should almost never be called directly.
     * Use instead @ref GaussianRBFKernel(const std::shared_ptr<Eigen::MatrixXd> &coord_mat_ptr, const ScaleMethod &method, const std::shared_ptr<Model> &model_ptr).
     */
    GaussianRBFKernel() {}

    /**
     * @brief Construct a new GaussianRBFKernel object.
     *
     * @param coord_mat_ptr Shared pointer to the matrix of particles' coordinates.
     * @param method Desired scale computation method.
     * @param model_ptr Shared pointer to the model object; only required for the @ref ScaleMethod::Hessian method.
     */
    GaussianRBFKernel(const std::shared_ptr<Eigen::MatrixXd> &coord_mat_ptr,
                      const ScaleMethod &method = ScaleMethod::Median,
                      const std::shared_ptr<Model> &model_ptr = nullptr)
        : Kernel(coord_mat_ptr->rows()),
          scale_method_(method),
          coord_matrix_ptr_(coord_mat_ptr),
          target_model_ptr_(model_ptr)
    {
        if (scale_method_ == ScaleMethod::Hessian && !model_ptr)
        {
            throw UnsetException("Hessian-based scale requires a model.");
        }

        if (dimension_ != coord_matrix_ptr_->rows())
        {
            throw DimensionMismatchException("The number of rows = " + std::to_string(coord_matrix_ptr_->rows()) + "in the provided coordinate matrix needs to match the specified dimension.");
        }

        // Initialize sizes
        pairwise_dist_vec_.resize(coord_matrix_ptr_->cols() * coord_matrix_ptr_->cols()); // add one more to consider self distance (which is zero) for convenience
        replicated_diag_matrix_.resize(coord_matrix_ptr_->cols(), coord_matrix_ptr_->cols());
        squared_coord_matrix_.resize(coord_matrix_ptr_->cols(), coord_matrix_ptr_->cols());
        squared_pairwise_dist_matrix_.resize(coord_matrix_ptr_->cols(), coord_matrix_ptr_->cols());

        kernel_parameters_.resize(1); // we use only the inverse matrix as the kernel parameter;
                                      // this can be optimized away if you don't intend to compose kernels from this class

        // Setup kernel function
        auto kernel_fun =
            [](const VectorXADd &x, const std::vector<MatrixXADd> &params, const VectorXADd &location) -> VectorXADd
        {
            VectorXADd result(1), diff = x - location;
            result << (-diff.transpose() * params[0] * diff).array().exp();
            return result;
        };

        // Compute initial scale
        UpdateParameters({ComputeScale()}); // compute the initial scale

        // Update kernel function and initialize
        UpdateKernel(kernel_fun);
    }

    /**
     * @brief Default destructor.
     *
     */
    ~GaussianRBFKernel() {}

    /**
     * @brief Assignment operator.
     */
    GaussianRBFKernel &operator=(const GaussianRBFKernel &obj)
    {
        scale_method_ = obj.scale_method_;
        coord_matrix_ptr_ = obj.coord_matrix_ptr_;
        target_model_ptr_ = obj.target_model_ptr_;

        Kernel::operator=(obj);

        return *this;
    }

    /**
     * @brief Prohibit implicit conversion by assignment operator.
     *
     */
    GaussianRBFKernel &operator=(const Kernel &obj) = delete;

    /**
     * @brief Copy an instance of this object into a unique pointer.
     *
     * @return Unique pointer to a copy of *this.
     */
    virtual std::unique_ptr<Kernel> CloneUniquePointer() const override
    {
        return std::make_unique<GaussianRBFKernel>(*this);
    }

    /**
     * @brief Copy an instance of this object into a shared pointer.
     *
     * @return Shared pointer to a copy of *this.
     */
    virtual std::shared_ptr<Kernel> CloneSharedPointer() const override
    {
        return std::make_shared<GaussianRBFKernel>(*this);
    }

    /**
     * @brief Execute methods required for each step of the SVGD.
     *
     * @details An override is provided to compute the scale each time this is called.
     */
    void Step() override
    {
        // Assign the same inverse scale matrix to each kernel parameter (will only have > 1 element if this is a composed kernel instance)
        Eigen::MatrixXd inv_scale_mat = ComputeScale();

        std::vector<Eigen::MatrixXd> params(kernel_parameters_.size());

        for (auto &param : params)
        {
            param = inv_scale_mat;
        }

        UpdateParameters(params);

        Initialize();
    }

protected:
    /**
     * @brief Compute the scale parameter of the kernel.
     *
     * @return The computed inverse scale matrix.
     */
    Eigen::MatrixXd ComputeScale()
    {
        switch (scale_method_)
        {
        case ScaleMethod::Median:
        {
            /*
                Heuristic obtained from: Q. Liu and D. Wang, “Stein Variational Gradient Descent: A General Purpose
                Bayesian Inference Algorithm,” in Advances in Neural Information Processing Systems, Curran Associates,
                Inc., 2016.

                For this method, (*coord_matrix_ptr_) should contain the m-dimensional coordinates of n particles as an (m x n) matrix
            */

            // Compute pairwise distances
            squared_coord_matrix_ = coord_matrix_ptr_->transpose() * (*coord_matrix_ptr_);

            replicated_diag_matrix_ = squared_coord_matrix_.diagonal().replicate(1, coord_matrix_ptr_->cols());

            squared_pairwise_dist_matrix_ = replicated_diag_matrix_ + replicated_diag_matrix_.transpose() - 2 * squared_coord_matrix_;

            pairwise_dist_vec_ = Eigen::Map<Eigen::RowVectorXd>(squared_pairwise_dist_matrix_.data(), coord_matrix_ptr_->cols() * coord_matrix_ptr_->cols()).array().sqrt();

            return std::log(coord_matrix_ptr_->cols()) / std::pow(ComputeMedian(pairwise_dist_vec_), 2) * Eigen::MatrixXd::Identity(dimension_, dimension_);
        }
        case ScaleMethod::Hessian:
        {
            /*
                Heuristic obtained from: G. Detommaso, T. Cui, Y. Marzouk, A. Spantini, and R. Scheichl, “A Stein
                variational Newton method,” in Advances in Neural Information Processing Systems, Curran Associates, Inc.,
                2018.

                For this method, (*coord_matrix_ptr_) should contain the m-dimensional coordinates of n particles as an (m x n) matrix
            */

            // Sum the hessians based on all of the particles' positions
            Eigen::MatrixXd hessian_sum(dimension_, dimension_);

            hessian_sum.setZero();

            for (int i = 0; i < coord_matrix_ptr_->cols(); ++i)
            {
                hessian_sum += -target_model_ptr_->EvaluateLogModelHessian(coord_matrix_ptr_->col(i));
            }

            return 1.0 / (2.0 * dimension_ * coord_matrix_ptr_->cols()) * hessian_sum;
        }
        default:
            throw std::invalid_argument(SVGDCPP_LOG_PREFIX + "[Argument error] Invalid scale method Enum provided.");
        }
    }

    /**
     * @brief Compute the median.
     *
     * @param row_vector Vector of values.
     * @return Median value.
     */
    double ComputeMedian(Eigen::RowVectorXd &row_vector)
    {
        if (row_vector.size() % 2 == 0) // even number of elements
        {
            /*
                To extract values a and b (the 2 middle values in a sorted array), we first get b by doing a partial
                sort (using nth_element) of row_vector and grabbing the (n/2 + 1)-th element.
            */

            size_t n_over_2_index = row_vector.size() / 2;

            std::nth_element(row_vector.data(), row_vector.data() + n_over_2_index, row_vector.data() + row_vector.size()); // this also sorts row_vector

            double b = row_vector(n_over_2_index);

            /*
                Due to the way nth_element works (partial sorting), the top half of row_vector is collectively >= the
                bottom half, so to get the (n/2)-th value we just need to get the largest number in the bottom half.
            */

            double a = *std::max_element(row_vector.data(), row_vector.data() + n_over_2_index);

            return (a + b) / 2.0;
        }
        else // odd number of elements
        {
            size_t n_minus_1_over_2 = row_vector.size() / 2; // integer division rounds down

            std::nth_element(row_vector.data(), row_vector.data() + n_minus_1_over_2, row_vector.data() + row_vector.size()); // this also sorts row_vector

            return row_vector(n_minus_1_over_2);
        }
    }

    Eigen::MatrixXd squared_coord_matrix_; ///< Squared coordinate matrix (coordinate matrix multiplied by its transposed).

    Eigen::MatrixXd squared_pairwise_dist_matrix_; ///< Squared pairwise distance matrix.

    Eigen::MatrixXd replicated_diag_matrix_; ///< Replicated diagonal matrix.

    Eigen::RowVectorXd pairwise_dist_vec_; ///< Pairwise distance row vector.

    ScaleMethod scale_method_; ///< Method used to compute the scale.

    std::shared_ptr<Eigen::MatrixXd> coord_matrix_ptr_; ///< Pointer to the particle coordinate matrix.

    std::shared_ptr<Model> target_model_ptr_; ///< Pointer to the model (or its derived class) object.
};

#endif
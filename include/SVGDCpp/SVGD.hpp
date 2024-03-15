#ifndef SVGD_CPP_SVGD_HPP
#define SVGD_CPP_SVGD_HPP

#include "Core.hpp"
#include "Kernel/Kernel.hpp"
#include "Model/Model.hpp"
#include "Optimizer/Optimizer.hpp"

// Particles struct
template <typename KType, typename MType, typename OType>
class SVGD
{
public:
    SVGD(
        const size_t &dim,
        const size_t &iter,
        const std::shared_ptr<Eigen::MatrixXd> &coord_mat_ptr,
        const std::shared_ptr<KType> &kernel_ptr,
        const std::shared_ptr<MType> &model_ptr,
        const std::shared_ptr<OType> &optimizer_ptr,
        const bool &parallel = false)
        : dimension_(coord_mat_ptr->rows()),
          num_iterations_(iter),
          parallel_(parallel)
    {
        if (dimension_ != dim)
        {
            throw std::runtime_error("Specified dimension does not match the particle coordinate matrix.");
        }

        coord_matrix_ptr_ = coord_mat_ptr;

        log_pdf_grad_matrix_.resize(dimension_, coord_matrix_ptr_->cols());                                               // m x n
        kernel_matrix_.resize(coord_matrix_ptr_->cols(), coord_matrix_ptr_->cols());                                      // n x n
        kernel_grad_matrix_.resize(dimension_ * coord_matrix_ptr_->cols(), coord_matrix_ptr_->cols());                    // (m*n) x n
        kernel_grad_indexer_ = Eigen::MatrixXd::Identity(dimension_, dimension_).replicate(1, coord_matrix_ptr_->cols()); // m x (m*n)

        // Store the kernel, distribution, and optimizer objects
        kernel_ptr_ = kernel_ptr;
        model_ptr_ = model_ptr;
        optimizer_ptr_ = optimizer_ptr;
    }

    ~SVGD() {}

    void Initialize()
    {
        // Initialize the model
        model_ptr_->Initialize();

        // Initialize the optimizer
        optimizer_ptr_->Initialize();

        // Initialize the kernel
        if (parallel_)
        {
            // Create n copies of the kernel function and initialize them
            kernel_vector_.resize(coord_matrix_ptr_->cols(), 1);

            for (size_t i = 0; i < coord_matrix_ptr_->cols(); ++i)
            {
                kernel_vector_(i) = *kernel_ptr_;
                kernel_vector_(i).UpdateLocation(coord_matrix_ptr_->col(i)); // particle coordinates matrix is expected to have dimension rows x n columns
                kernel_vector_(i).Initialize();
            }
        }
        else
        {
            kernel_ptr_->Initialize();
        }
    }

    void UpdateKernel(const std::vector<Eigen::MatrixXd> &params)
    {
        if (parallel_)
        {
            std::for_each(kernel_vector_.data().begin(),
                          kernel_vector_.data().end(),
                          [=](KType &k)
                          {
                              k.UpdateParameters(params);
                              k.Initialize();
                          });
        }
        else
        {
            kernel_ptr_->UpdateParameters(params);
        }
    }

    void UpdateModel(const std::vector<Eigen::MatrixXd> &params)
    {
        model_ptr_->UpdateParameters(params);
    }

    void Run()
    {
        // Run L iterations
        for (size_t iter = 0; iter < num_iterations_; ++iter)
        {
            // x + e * phi(x)
            Step();
        }
    }

protected:
    void Step()
    {
        model_ptr_->Step();

        kernel_ptr_->Step();

        // Update particle positions
        *coord_matrix_ptr_ += optimizer_ptr_->Step(ComputePhi());
    }

    /**
     * @brief Compute the Stein variational gradient
     *
     * @return Eigen::MatrixXd Matrix of gradients with the size of (dimension) x (num of particles)
     */
    Eigen::MatrixXd ComputePhi()
    {
        for (size_t i = 0; i < coord_matrix_ptr_->cols(); ++i)
        {
            // Compute log pdf grad
            log_pdf_grad_matrix_.block(0, i, dimension_, 1) = model_ptr_->EvaluateLogModelGrad(coord_matrix_ptr_->col(i));

            kernel_ptr_->UpdateLocation(coord_matrix_ptr_->col(i));

            // Compute kernel and grad kernel
            if (parallel_)
            {
                // TODO
                // // Update kernel function location
                // kernel_vector_(i).UpdateLocation(coord_matrix_ptr_->col(i));

                // // Run a single step on the kernel functions
                // kernel_vector_(i).Step();

                // for (size_t j = 0; j < coord_matrix_ptr_->cols(); ++i)
                // {
                //     kernel_val(i, j) = kernel_vector_(i).EvaluateKernel(coord_matrix_ptr_->col(j));                                            // k(x_j, x_i)
                //     kernel_grad_val.block(j * dimension_, i, dimension_, 1) = kernel_vector_(i).EvaluateKernelGrad(coord_matrix_ptr_->col(j)); // grad k(x_j, x_i)
                // }
            }
            else
            {
                for (size_t j = 0; j < coord_matrix_ptr_->cols(); ++j)
                {
                    kernel_matrix_(j, i) = kernel_ptr_->EvaluateKernel(coord_matrix_ptr_->col(j));                                            // k(x_j, x_i)
                    kernel_grad_matrix_.block(j * dimension_, i, dimension_, 1) = kernel_ptr_->EvaluateKernelGrad(coord_matrix_ptr_->col(j)); // grad k(x_j, x_i)
                }
            }
        }

        return (1.0 / coord_matrix_ptr_->cols()) * (log_pdf_grad_matrix_ * kernel_matrix_ + kernel_grad_indexer_ * kernel_grad_matrix_);
    }

    size_t dimension_;

    size_t num_iterations_;

    double step_size_;

    const bool parallel_;

    Eigen::MatrixXd log_pdf_grad_matrix_;

    Eigen::MatrixXd kernel_matrix_;

    Eigen::MatrixXd kernel_grad_matrix_;

    Eigen::MatrixXd kernel_grad_indexer_;

    Eigen::Matrix<KType, -1, 1> kernel_vector_;

    std::shared_ptr<Eigen::MatrixXd> coord_matrix_ptr_;

    // Idea: we store N kernel function objects for N corresponding particles; the particles' state is used to parametrize
    // the kernels. Here we're trying to trade memory for speed; with a kernel 'responsible' for each particle we don't
    // need to keep updating the kernel parameters
    std::shared_ptr<KType> kernel_ptr_;

    std::shared_ptr<MType> model_ptr_;

    std::shared_ptr<OType> optimizer_ptr_;
};

#endif
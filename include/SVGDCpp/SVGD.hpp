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
    SVGD(const size_t dim, const size_t &iter, const bool &parallel = false)
        : dimension_(dim),
          num_iterations_(iter),
          parallel_(parallel),
          kernel_ptr_(nullptr),
          model_ptr_(nullptr),
          optimizer_ptr_(nullptr)
    {
    }

    ~SVGD() {}

    void Initialize(const std::shared_ptr<Particles> &particles_ptr,
                    const std::shared_ptr<KType> &kernel_ptr,
                    const std::shared_ptr<MType> &model_ptr,
                    const std::shared_ptr<OType> &optimizer_ptr)
    {
        particles_ptr_ = particles_ptr;

        // // Create n copies of the kernel function to store
        // for (size_t i = 0; i < particles_ptr_->n; ++i)
        // {
        //     KType k = kernel_obj;
        //     k.Initialize();
        //     kernel_vec_ptr_->push_back(k);
        // }

        log_pdf_grad_matrix_.resize(particles_ptr_->n, dimension_);                                              // n x m
        kernel_matrix_.resize(particles_ptr_->n, particles_ptr_->n);                                             // n x n
        kernel_grad_matrix_.resize(dimension_ * particles_ptr_->n, particles_ptr_->n);                           // (m*n) x n
        kernel_grad_indexer_ = Eigen::MatrixXd::Identity(dimension_, dimension_).replicate(particles_ptr->n, 1); // (m*n) x m

        // Store the kernel, distribution, and optimzer objects
        kernel_ptr_ = kernel_ptr;
        model_ptr_ = model_ptr;
        optimizer_ptr_ = optimizer_ptr;

        if (parallel_)
        {
            // Create n copies of the kernel function and initialize them
            kernel_vector_.resize(particles_ptr_->n, 1);

            for (size_t i = 0; i < particles_ptr_->n; ++i)
            {
                kernel_vector_(i) = *kernel_ptr_;
                kernel_vector_(i).UpdateLocation(particles_ptr_->coordinates.col(i)); // particle coordinates matrix is expected to have dimension rows x n columns
                kernel_vector_(i).Initialize();
            }
        }

        // Initialize the distribution
        model_ptr_->Initialize();
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

    std::vector<Eigen::VectorXd> Run()
    {
        // Run L iterations
        for (size_t i = 0; i < num_iterations_; ++i)
        {
            // x + e * phi(x)
            Step();
        }

        std::vector<Eigen::VectorXd> vec(particles_ptr_->n);

        for (size_t i = 0; i < particles_ptr_->n; ++i)
        {
            vec[i] = particles_ptr_->coordinates.col(i);
        }

        return vec;
    }

protected:
    void Step()
    {
        model_ptr_->Step();

        kernel_ptr_->Step();

        // Update particle positions
        particles_ptr_->coordinates += optimizer_ptr_->Step(ComputePhi());
    }

    /**
     * @brief Compute the Stein variational gradient
     *
     * @return Eigen::MatrixXd Matrix of gradients with the size of (dimension) x (num of particles)
     */
    Eigen::MatrixXd ComputePhi()
    {

        for (size_t i = 0; i < particles_ptr_->n; ++i)
        {
            // Compute log pdf grad
            log_pdf_grad_matrix_.block(i, 0, 1, dimension_) = model_ptr_->EvaluateLogModelGrad(particles_ptr_->coordinates.col(i));

            // Compute kernel and grad kernel
            if (parallel_)
            {
                // TODO
                // // Update kernel function location
                // kernel_vector_(i).UpdateLocation(particles_ptr_->coordinates.col(i));

                // // Run a single step on the kernel functions
                // kernel_vector_(i).Step();

                // for (size_t j = 0; j < particles_ptr_->n; ++i)
                // {
                //     kernel_val(i, j) = kernel_vector_(i).EvaluateKernel(particles_ptr_->coordinates.col(j));                                            // k(x_j, x_i)
                //     kernel_grad_val.block(j * dimension_, i, dimension_, 1) = kernel_vector_(i).EvaluateKernelGrad(particles_ptr_->coordinates.col(j)); // grad k(x_j, x_i)
                // }
            }
            else
            {
                for (size_t j = 0; j < particles_ptr_->n; ++i)
                {
                    kernel_matrix_(i, j) = kernel_ptr_->EvaluateKernel(particles_ptr_->coordinates.col(j));                                            // k(x_j, x_i)
                    kernel_grad_matrix_.block(j * dimension_, i, dimension_, 1) = kernel_ptr_->EvaluateKernelGrad(particles_ptr_->coordinates.col(j)); // grad k(x_j, x_i)
                }
            }
        }

        return (1.0 / particles_ptr_->n) * (kernel_matrix_ * log_pdf_grad_matrix_ + kernel_grad_matrix_.transpose() * kernel_grad_indexer_).transpose();
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

    std::shared_ptr<Particles> particles_ptr_;

    // Idea: we store N kernel function objects for N corresponding particles; the particles' state is used to parametrize
    // the kernels. Here we're trying to trade memory for speed; with a kernel 'responsible' for each particle we don't
    // need to keep updating the kernel parameters
    std::shared_ptr<KType> kernel_ptr_;

    std::shared_ptr<MType> model_ptr_;

    std::shared_ptr<OType> optimizer_ptr_;
};

#endif
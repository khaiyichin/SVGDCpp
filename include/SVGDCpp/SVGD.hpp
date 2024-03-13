#ifndef SVGD_HPP
#define SVGD_HPP

#include "SVGDCppCore.hpp"
#include "Distribution/Distribution.hpp"

// Particles struct
template <typename KType, typename DType>
class SVGD
{
public:
    SVGD(const size_t dim, const size_t &iter, const bool &parallel = false)
        : dimension_(dim),
          num_iterations_(iter),
          parallel_(parallel),
          //   kernel_vec_ptr_(std::make_unique<std::vector<KType>>()),
          kernel_ptr_(std::make_unique<KType>()),
          distribution_ptr_(std::make_unique<DType>())
    {
    }

    ~SVGD() {}

    void test()
    {
        std::cout << "debug " << distribution_ptr_->GetNormConst() << std::endl;
    }

    void Initialize(const std::shared_ptr<Particles> &particles_ptr, const KType &kernel_obj, const DType &distribution_obj)
    {
        particles_ptr_ = particles_ptr;

        // // Create n copies of the kernel function to store
        // for (size_t i = 0; i < particles_ptr_->n; ++i)
        // {
        //     KType k = kernel_obj;
        //     k.Initialize();
        //     kernel_vec_ptr_->push_back(k);
        // }

        kernel_grad_indexer_ = Eigen::MatrixXd::Identity(dimension_, dimension_).replicate(particles_ptr->n, 1);

        // Store the kernel and distribution objects
        kernel_ptr_ = std::make_unique<KType>(kernel_obj); // CAN WE REMOVE THIS???? since we store the kernel in the eigen matrix already, what's the point of having this unique pointer around?
        distribution_ptr_ = std::make_shared<DType>(distribution_obj);

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
        distribution_ptr_->Initialize();
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

    void UpdateDistribution(const std::vector<Eigen::MatrixXd> &params)
    {
        distribution_ptr_->UpdateParameters(params);
    }

    std::vector<Eigen::VectorXd> Run(const std::vector<Eigen::VectorXd> &particle_coords)
    {

        // Eigen::Map<Eigen::Matrix<KType, 1, -1>> kernel_matrix(kernel_vec_ptr_->data(), 1, particles_ptr_->n);

        // Create n copies of the kernel function
        // Eigen::Matrix<KType, 1, -1> kernel_matrix(particles_ptr_->n);

        // for (size_t i = 0; i < particles_ptr_->n; ++i)
        // {
        //     kernel_matrix(i) = *kernel_ptr_;
        // }

        // Run L iterations
        for (size_t i = 0; i < num_iterations_; ++i)
        {
            // x + e * phi(x)
            Step();
        }
    }

protected:
    // compute phi

    void Step()
    {
        Eigen::MatrixXd phi_matrix = ComputePhi();

        // Compute step size
        // TODO
        Eigen::MatrixXd step_matrix = AdjustStepSize(phi_matrix);

        // Update particle positions
        particles_ptr_->coordinates += step_matrix;
    }

    Eigen::MatrixXd ComputePhi()
    {
        // Compute log pdf grad, kernel and kernel grad
        Eigen::MatrixXd log_pdf_grad_val(particles_ptr_->n, dimension_);                    // n x m
        Eigen::MatrixXd kernel_val(particles_ptr_->n, particles_ptr_->n);                   // n x n
        Eigen::MatrixXd kernel_grad_val(dimension_ * particles_ptr_->n, particles_ptr_->n); // (nxm) x n

        distribution_ptr_->Step();

        for (size_t i = 0; i < particles_ptr_->n; ++i)
        {
            // Compute log pdf grad
            log_pdf_grad_val.block(i, 0, 1, dimension_) = distribution_ptr_->GetLogPDFGrad(particles_ptr_->coordinates.col(i));

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
                //     kernel_val(i, j) = kernel_vector_(i).GetKernel(particles_ptr_->coordinates.col(j));                                            // k(x_j, x_i)
                //     kernel_grad_val.block(j * dimension_, i, dimension_, 1) = kernel_vector_(i).GetKernelGrad(particles_ptr_->coordinates.col(j)); // grad k(x_j, x_i)
                // }
            }
            else
            {
                for (size_t j = 0; j < particles_ptr_->n; ++i)
                {
                    kernel_val(i, j) = kernel_ptr_->GetKernel(particles_ptr_->coordinates.col(j));                                               // k(x_j, x_i)
                    kernel_grad_val.block(j * dimension_, i, dimension_, 1) = kernel_ptr_->GetKernelGrad(particles_ptr_->coordinates.col(j)); // grad k(x_j, x_i)
                }
            }
        }

        // Compute Phi matrix
        return 1.0 / particles_ptr_->n * (kernel_val * log_pdf_grad_val + kernel_grad_val.transpose() * kernel_grad_indexer_).transpose();
    }

    Eigen::MatrixXd AdjustStepSize(const Eigen::MatrixXd &phi)
    {
        // TODO
    }

    size_t dimension_;

    size_t num_iterations_;

    double step_size_;

    const bool parallel_;

    Eigen::MatrixXd kernel_grad_indexer_;

    Eigen::Matrix<KType, -1, 1> kernel_vector_;

    std::shared_ptr<Particles> particles_ptr_;

    std::shared_ptr<DType> distribution_ptr_;

    // Idea: we store N kernel function objects for N corresponding particles; the particles' state is used to parametrize
    // the kernels. Here we're trying to trade memory for speed; with a kernel 'responsible' for each particle we don't
    // need to keep updating the kernel parameters
    // std::unique_ptr<std::vector<KType>> kernel_vec_ptr_;
    std::unique_ptr<KType> kernel_ptr_;
};

#endif
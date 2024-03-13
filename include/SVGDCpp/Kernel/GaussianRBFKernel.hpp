#ifndef GAUSSIAN_RBF_KERNEL_HPP
#define GAUSSIAN_RBF_KERNEL_HPP

#include "../SVGDCppCore.hpp"
#include "../Distribution/Distribution.hpp"
#include "KernelFun.hpp"

class GaussianRBFKernel : public KernelFun
{
public:
    enum class ScaleMethod
    {
        Median = 0,
        Hessian = 1
    };

    GaussianRBFKernel() {}

    GaussianRBFKernel(const GaussianRBFKernel &obj)
    {
        *this = obj;
    }

    GaussianRBFKernel(const std::shared_ptr<Particles> p_ptr,
                      const ScaleMethod &method = ScaleMethod::Median,
                      const std::shared_ptr<Distribution> &dist_ptr = nullptr)
        : KernelFun(p_ptr->coordinates.rows()),
          particles_ptr_(p_ptr),
          scale_method_(method),
          target_distribution_ptr_(dist_ptr)
    {
        ComputeScale();
    }

    ~GaussianRBFKernel() {}

    GaussianRBFKernel &operator=(const GaussianRBFKernel &obj)
    {
        scale_mat_ad_ = obj.scale_mat_ad_;
        scale_method_ = obj.scale_method_;
        particles_ptr_ = obj.particles_ptr_;
        target_distribution_ptr_ = obj.target_distribution_ptr_;

        KernelFun::operator=(obj);

        return *this;
    }

    /**
     * @brief
     * The params argument is not used in this method for this class
     *
     * @param params
     */
    void UpdateParameters(const std::vector<Eigen::MatrixXd> &params = std::vector<Eigen::MatrixXd>()) override
    {
        ComputeScale();
    }

    void Step() override
    {
        ComputeScale();

        SetupADFun();
    }

protected:
    VectorXADd Kernel(const VectorXADd &x) override
    {
        VectorXADd diff = x - location_vec_ad_;
        return (-diff.transpose() * scale_mat_ad_ * diff).array().exp();
    }

    void ComputeScale()
    {
        switch (scale_method_)
        {
        case ScaleMethod::Median:
        {
            /*
                Heuristic obtained from: Q. Liu and D. Wang, “Stein Variational Gradient Descent: A General Purpose
                Bayesian Inference Algorithm,” in Advances in Neural Information Processing Systems, Curran Associates,
                Inc., 2016.

                For this method, particles_ptr_->coordinates should contain the m-dimensional coordinates of n particles as an (m x n) matrix
            */

            // Create location vector of type double to perform arithmetic
            Eigen::VectorXd location_vec(location_vec_ad_.size());

            for (size_t i = 0; i < location_vec_ad_.size(); ++i)
            {
                location_vec(i) = CppAD::Value(location_vec_ad_(i));
            }

            // Compute pairwise distances from current location coordinate
            Eigen::RowVectorXd distances = (particles_ptr_->coordinates.colwise() - location_vec).colwise().squaredNorm();

            // Compute the scale
            scale_mat_ad_ =
                Eigen::Matrix<CppAD::AD<double>, 1, 1>(std::pow(ComputeMedian(distances), 2) / std::log(particles_ptr_->n));

            break;
        }
        case ScaleMethod::Hessian:
        {
            /*
                Heuristic obtained from: G. Detommaso, T. Cui, Y. Marzouk, A. Spantini, and R. Scheichl, “A Stein
                variational Newton method,” in Advances in Neural Information Processing Systems, Curran Associates, Inc.,
                2018.

                For this method, particles_ptr_->coordinates should contain the m-dimensional coordinates of n particles as an (m x n) matrix
            */

            target_distribution_ptr_->Initialize();

            // Sum the hessians based on all of the particles' positions
            Eigen::MatrixXd hessian_sum(dimension_, dimension_);

            hessian_sum.setZero();

            for (size_t i = 0; i < particles_ptr_->n; ++i)
            {
                hessian_sum += target_distribution_ptr_->GetLogPDFHessian(particles_ptr_->coordinates.col(i));
            }

            scale_mat_ad_ = (1.0 / (2.0 * particles_ptr_->n * dimension_) * hessian_sum).cast<CppAD::AD<double>>();

            break;
        }
        }
    }

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

    MatrixXADd scale_mat_ad_;

    ScaleMethod scale_method_;

    std::shared_ptr<Particles> particles_ptr_;

    std::shared_ptr<Distribution> target_distribution_ptr_;
};

#endif
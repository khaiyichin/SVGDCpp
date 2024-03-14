#ifndef SVGD_CPP_KERNEL_HPP
#define SVGD_CPP_KERNEL_HPP

#include "../Core.hpp"

class Kernel
{
public:
    Kernel() {}

    virtual ~Kernel(){};

    Kernel(const Kernel &obj)
    {
        dimension_ = obj.dimension_;
        kernel_ad_ = obj.kernel_ad_;
    }

    Kernel(const size_t &dim) : dimension_(dim), location_vec_ad_(dim) {}

    virtual Kernel &operator=(const Kernel &obj)
    {
        dimension_ = obj.dimension_;
        location_vec_ad_ = obj.location_vec_ad_;
        kernel_ad_ = obj.kernel_ad_;

        return *this;
    }

    virtual void Initialize()
    {
        SetupADFun();
    }

    double GetKernel(const Eigen::VectorXd &x)
    {
        return kernel_ad_.Forward(0, x)(0, 0);
    }

    Eigen::VectorXd GetKernelGrad(const Eigen::VectorXd &x)
    {
        return kernel_ad_.Jacobian(x);
    }

    /**
     * @brief Update the distribution with new parameters
     *
     * @param params Vector of parameters in Eigen::VectorXd or Eigen::MatrixXd
     */
    virtual void Update(const std::vector<Eigen::MatrixXd> &params)
    {
        UpdateParameters(params);

        SetupADFun();
    }

    virtual void UpdateParameters(const std::vector<Eigen::MatrixXd> &params) = 0;

    virtual void UpdateLocation(const Eigen::VectorXd &x) { location_vec_ad_ = x.cast<CppAD::AD<double>>(); }

    virtual void Step()
    {
        SetupADFun();
    }

protected:
    virtual VectorXADd KernelFun(const VectorXADd &x) = 0;

    virtual void SetupADFun()
    {
        VectorXADd x_kernel_ad(dimension_), y_kernel_ad(dimension_);

        // Setup PDF
        CppAD::Independent(x_kernel_ad); // start recording sequence

        y_kernel_ad = KernelFun(x_kernel_ad);

        kernel_ad_ = CppAD::ADFun<double>(x_kernel_ad, y_kernel_ad); // store operation sequence and stop recording
    }

    size_t dimension_;

    VectorXADd location_vec_ad_;

    CppAD::ADFun<double> kernel_ad_;
};

#endif
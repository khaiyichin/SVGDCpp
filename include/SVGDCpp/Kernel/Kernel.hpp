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
        *this = obj;
    }

    Kernel(const size_t &dim) : dimension_(dim), location_vec_ad_(dim) {}

    virtual Kernel &operator=(const Kernel &obj)
    {
        dimension_ = obj.dimension_;
        location_vec_ad_ = obj.location_vec_ad_;
        kernel_fun_ad_ = obj.kernel_fun_ad_;

        return *this;
    }

    virtual void Initialize()
    {
        SetupADFun();
    }

    double EvaluateKernel(const Eigen::VectorXd &x)
    {
        return kernel_fun_ad_.Forward(0, x)(0, 0);
    }

    Eigen::VectorXd EvaluateKernelGrad(const Eigen::VectorXd &x)
    {
        return kernel_fun_ad_.Jacobian(x);
    }

    virtual void UpdateParameters(const std::vector<Eigen::MatrixXd> &params) = 0;

    virtual void UpdateLocation(const Eigen::VectorXd &x)
    {
        location_vec_ad_ = x.cast<CppAD::AD<double>>();

        Initialize();
    }

    virtual void Step() {}

protected:
    virtual VectorXADd KernelFun(const VectorXADd &x) = 0;

    virtual void SetupADFun()
    {
        VectorXADd x_kernel_ad(dimension_), y_kernel_ad(dimension_);

        // Setup PDF
        CppAD::Independent(x_kernel_ad); // start recording sequence

        y_kernel_ad = KernelFun(x_kernel_ad);

        kernel_fun_ad_ = CppAD::ADFun<double>(x_kernel_ad, y_kernel_ad); // store operation sequence and stop recording
    }

    size_t dimension_;

    VectorXADd location_vec_ad_;

    CppAD::ADFun<double> kernel_fun_ad_;
};

#endif
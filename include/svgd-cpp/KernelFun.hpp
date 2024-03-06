#ifndef KERNEL_FUN_HPP
#define KERNEL_FUN_HPP

#include "SVGDCppCore.hpp"

class KernelFun
{
public:
    KernelFun(const size_t &dim) : dimension_(dim) {}

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
    void Update(const std::vector<Eigen::MatrixXd> &params)
    {
        UpdateParameters(params);

        SetupADFun();
    }

protected:
    virtual VectorXADd Kernel(const VectorXADd &x) = 0;

    virtual void UpdateParameters(const std::vector<Eigen::MatrixXd> &params) = 0;

    virtual void SetupADFun()
    {
        VectorXADd x_kernel_ad(dimension_), y_kernel_ad(dimension_);

        // Setup PDF
        CppAD::Independent(x_kernel_ad); // start recording sequence

        y_kernel_ad = Kernel(x_kernel_ad);

        kernel_ad_ = CppAD::ADFun<double>(x_kernel_ad, y_kernel_ad); // store operation sequence and stop recording
    }

    size_t dimension_;

    CppAD::ADFun<double> kernel_ad_;
};

#endif
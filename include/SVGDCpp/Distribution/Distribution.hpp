#ifndef SVGD_CPP_DISTRIBUTION_HPP
#define SVGD_CPP_DISTRIBUTION_HPP

#include "../Core.hpp"
#include "../Kernel/Kernel.hpp"

class Distribution : public Kernel
{
public:
    Distribution() {}

    Distribution(const size_t &dim) : Kernel(dim) {}

    virtual ~Distribution() {}

    double GetPDF(const Eigen::VectorXd &x)
    {
        return pdf_ad_.Forward(0, x)(0, 0);
    }

    double GetLogPDF(const Eigen::VectorXd &x)
    {
        return logpdf_ad_.Forward(0, x)(0, 0);
    }

    Eigen::VectorXd GetPDFGrad(const Eigen::VectorXd &x)
    {
        return pdf_ad_.Jacobian(x);
    }

    Eigen::VectorXd GetLogPDFGrad(const Eigen::VectorXd &x)
    {
        return logpdf_ad_.Jacobian(x);
    }

    Eigen::MatrixXd GetPDFHessian(const Eigen::VectorXd &x)
    {
        // PDF is a scalar function (1-D output), so evaluating Hessian at function of index 0
        return Eigen::Map<Eigen::Matrix<double, -1, -1>>(pdf_ad_.Hessian(x, 0).data(), dimension_, dimension_).transpose(); // need to transpose because of column-major (default) storage
    }

    Eigen::MatrixXd GetLogPDFHessian(const Eigen::VectorXd &x)
    {
        // PDF is a scalar function (1-D output), so evaluating Hessian at function of index 0
        return Eigen::Map<Eigen::Matrix<double, -1, -1>>(logpdf_ad_.Hessian(x, 0).data(), dimension_, dimension_).transpose(); // need to transpose because of column-major (default) storage
    }

    double GetNormConst() { return norm_const_; }

    virtual Distribution &operator=(const Distribution &obj)
    {
        pdf_ad_ = obj.pdf_ad_;
        logpdf_ad_ = obj.logpdf_ad_;
        norm_const_ = obj.norm_const_;

        Kernel::operator=(obj);

        return *this;
    }

    virtual void Step() = 0;

protected:
    virtual VectorXADd PDF(const VectorXADd &x)
    {
        return norm_const_ * KernelFun(x);
    }

    virtual VectorXADd LogPDF(const VectorXADd &x)
    {
        return std::log(norm_const_) + KernelFun(x).array().log();
    }

    /**
     * @brief Setup PDF and LogPDF functions, to be called everytime the distribution parameters are updated
     *
     */
    virtual void SetupADFun() override
    {
        VectorXADd x_pdf_ad(dimension_), y_pdf_ad(dimension_),
            x_logpdf_ad(dimension_), y_logpdf_ad(dimension_);

        // Setup PDF
        CppAD::Independent(x_pdf_ad); // start recording sequence

        y_pdf_ad = PDF(x_pdf_ad);

        pdf_ad_ = CppAD::ADFun<double>(x_pdf_ad, y_pdf_ad); // store operation sequence and stop recording

        pdf_ad_.optimize();

        // Setup LogPDF
        CppAD::Independent(x_logpdf_ad); // start recording sequence

        y_logpdf_ad = LogPDF(x_logpdf_ad);

        logpdf_ad_ = CppAD::ADFun<double>(x_logpdf_ad, y_logpdf_ad); // store operation sequence and stop recording

        logpdf_ad_.optimize();
    }

    virtual void ComputeNormConst() = 0;

    CppAD::ADFun<double> pdf_ad_;

    CppAD::ADFun<double> logpdf_ad_;

    double norm_const_;
};

#endif
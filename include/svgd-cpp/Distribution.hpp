#ifndef DISTRIBUTION_HPP
#define DISTRIBUTION_HPP

#include "SVGDCppCore.hpp"
#include "KernelFun.hpp"

class Distribution : public KernelFun
{
public:
    Distribution(const size_t &dim) : KernelFun(dim) {}

    ~Distribution() {}

    Eigen::VectorXd GetPDFGrad(const Eigen::VectorXd &x)
    {
        return pdf_ad_.Jacobian(x);
    }

    Eigen::VectorXd GetLogPDFGrad(const Eigen::VectorXd &x)
    {
        return logpdf_ad_.Jacobian(x);
    }

    double GetPDF(const Eigen::VectorXd &x)
    {
        return pdf_ad_.Forward(0, x)(0, 0);
    }

    double GetLogPDF(const Eigen::VectorXd &x)
    {
        return logpdf_ad_.Forward(0, x)(0, 0);
    }

    double GetNormConst() { return norm_const_; }

protected:
    virtual VectorXADd PDF(const VectorXADd &x)
    {
        return norm_const_ * Kernel(x);
    }

    virtual VectorXADd LogPDF(const VectorXADd &x)
    {
        return PDF(x).array().log();
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

        // Setup LogPDF
        CppAD::Independent(x_logpdf_ad); // start recording sequence

        y_logpdf_ad = LogPDF(x_logpdf_ad);

        logpdf_ad_ = CppAD::ADFun<double>(x_logpdf_ad, y_logpdf_ad); // store operation sequence and stop recording
    }

    virtual void ComputeNormConst() = 0;

    CppAD::ADFun<double> pdf_ad_;

    CppAD::ADFun<double> logpdf_ad_;

    double norm_const_;
};

#endif
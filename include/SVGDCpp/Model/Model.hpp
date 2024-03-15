#ifndef SVGD_CPP_MODEL_HPP
#define SVGD_CPP_MODEL_HPP

#include "../Core.hpp"

class Model
{
public:
    Model() {}

    virtual ~Model(){};

    Model(const Model &obj)
    {
        *this = obj;
    }

    Model(const size_t &dim) : dimension_(dim) {}

    virtual Model &operator=(const Model &obj)
    {
        dimension_ = obj.dimension_;
        model_fun_ad_ = obj.model_fun_ad_;
        logmodel_fun_ad_ = obj.logmodel_fun_ad_;

        return *this;
    }

    virtual void Initialize()
    {
        SetupADFun();
    }

    double EvaluateModel(const Eigen::VectorXd &x)
    {
        return model_fun_ad_.Forward(0, x)(0, 0);
    }

    double EvaluateLogModel(const Eigen::VectorXd &x)
    {
        return logmodel_fun_ad_.Forward(0, x)(0, 0);
    }

    Eigen::VectorXd EvaluateModelGrad(const Eigen::VectorXd &x)
    {
        return model_fun_ad_.Jacobian(x);
    }

    Eigen::VectorXd EvaluateLogModelGrad(const Eigen::VectorXd &x)
    {
        return logmodel_fun_ad_.Jacobian(x);
    }

    Eigen::MatrixXd EvaluateModelHessian(const Eigen::VectorXd &x)
    {
        // Model is a scalar function (1-D output), so evaluating Hessian at function of index 0
        return Eigen::Map<Eigen::Matrix<double, -1, -1>>(model_fun_ad_.Hessian(x, 0).data(), dimension_, dimension_).transpose(); // need to transpose because of column-major (default) storage
    }

    Eigen::MatrixXd EvaluateLogModelHessian(const Eigen::VectorXd &x)
    {
        // Model is a scalar function (1-D output), so evaluating Hessian at function of index 0
        return Eigen::Map<Eigen::Matrix<double, -1, -1>>(logmodel_fun_ad_.Hessian(x, 0).data(), dimension_, dimension_).transpose(); // need to transpose because of column-major (default) storage
    }

    virtual void UpdateParameters(const std::vector<Eigen::MatrixXd> &params) = 0;

    virtual void Step() {}

protected:
    virtual VectorXADd ModelFun(const VectorXADd &x) = 0;

    virtual VectorXADd LogModelFun(const VectorXADd &x)
    {
        return ModelFun(x).array().log();
    }

    virtual void SetupADFun()
    {
        VectorXADd x_model_ad(dimension_), y_model_ad(dimension_),
            x_logmodel_ad(dimension_), y_logmodel_ad(dimension_);

        // Setup PDF
        CppAD::Independent(x_model_ad); // start recording sequence

        y_model_ad = ModelFun(x_model_ad);

        model_fun_ad_ = CppAD::ADFun<double>(x_model_ad, y_model_ad); // store operation sequence and stop recording

        model_fun_ad_.optimize();

        // Setup LogPDF
        CppAD::Independent(x_logmodel_ad); // start recording sequence

        y_logmodel_ad = LogModelFun(x_logmodel_ad);

        logmodel_fun_ad_ = CppAD::ADFun<double>(x_logmodel_ad, y_logmodel_ad); // store operation sequence and stop recording

        logmodel_fun_ad_.optimize();
    }

    size_t dimension_;

    CppAD::ADFun<double> model_fun_ad_;

    CppAD::ADFun<double> logmodel_fun_ad_;
};

#endif
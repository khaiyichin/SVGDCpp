/**
 * @file MultivariateNormal.hpp
 * @author Khai Yi Chin (khaiyichin@gmail.com)
 * @brief Multivariate normal model class header.
 * @version 0.1
 * @date 2024-03-22
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef SVGD_CPP_MULTIVARIATE_NORMAL_HPP
#define SVGD_CPP_MULTIVARIATE_NORMAL_HPP

#include "../Core.hpp"
#include "Model.hpp"

/**
 * @class MultivariateNormal
 * @brief Implementation of a Multivariate Normal model.
 * @details The base methods @ref Model::EvaluateModel and @ref Model::EvaluateLogModel are not normalized;
 * use @ref EvaluateModelNormalized and @ref EvaluateLogModelNormalized instead if normalized values are desired.
 * @ingroup Model_Module
 */
class MultivariateNormal : public Model
{
public:
    /**
     * @brief Default constructor.
     * @details This should almost never be called directly.
     * Use instead @ref MultivariateNormal(const Eigen::VectorXd &mean, const Eigen::MatrixXd &covariance).
     */
    MultivariateNormal() {}

    /**
     * @brief Construct a new MultivariateNormal object.
     * @details This is the preferred method to instantiate a Multivariate Normal class.
     * @param mean Variable-sized Eigen vector whose dimensions must be compatible with @a covariance.
     * @param covariance Variable-sized Eigen matrix whose dimensions must be compatible with @a mean.
     */
    MultivariateNormal(const Eigen::VectorXd &mean, const Eigen::MatrixXd &covariance) : Model(mean.rows())
    {
        // Ensure that the dimensions of mean matches covariance
        if (!CompareVectorSizes<Eigen::VectorXd, Eigen::VectorXd>(mean, covariance.col(0)) ||
            !CompareVectorSizes<Eigen::VectorXd, Eigen::VectorXd>(mean, covariance.row(0)))
        {
            throw DimensionMismatchException("Dimensions of parameter vectors/matrices do not match.");
        }

        // Store parameters
        model_parameters_.push_back(ConvertToCppAD(mean));
        model_parameters_.push_back(ConvertToCppAD(covariance));

        // Compute the normalization constant based on the updated parameters
        ComputeNormalizationConstant();

        // Define model function (the kernel density only, without normalization constant)
        auto model_fun = [this](const VectorXADd &x, const std::vector<MatrixXADd> &params)
        {
            VectorXADd result(1), diff = x - params[0];
            result << (-0.5 * (diff.transpose() * params[1].inverse() * diff).array()).exp();
            return result;
        };

        UpdateModel(model_fun);
    }

    /**
     * @brief Default destructor.
     *
     */
    ~MultivariateNormal() {}

    /**
     * @brief Assignment operator.
     */
    MultivariateNormal &operator=(const MultivariateNormal &obj)
    {
        Model::operator=(obj);
        norm_const_ = obj.norm_const_;

        return *this;
    }

    /**
     * @brief Prohibit implicit conversion by assignment operator.
     *
     */
    MultivariateNormal &operator=(const Model &other) = delete;

    /**
     * @brief Update the model parameters.
     * @details This method is overridden to provide some safeguards but is otherwise identical to @ref Model::UpdateParameters.
     * @param params Vector of variable-sized Eigen objects.
     */
    void UpdateParameters(const std::vector<Eigen::MatrixXd> &params) override
    {
        Eigen::VectorXd mean = params[0];
        Eigen::MatrixXd covariance = params[1];

        // Ensure that the dimensions of mean matches covariance
        if (!CompareVectorSizes<Eigen::VectorXd, Eigen::VectorXd>(mean, covariance.col(0)) ||
            !CompareVectorSizes<Eigen::VectorXd, Eigen::VectorXd>(mean, covariance.row(0)))
        {
            throw DimensionMismatchException("Dimensions of parameter vectors/matrices do not match each other (# of rows must be equal).");
        }
        else if (mean.rows() != dimension_)
        {
            throw DimensionMismatchException("Dimensions of parameter vectors/matrices do not match original dimension.");
        }

        model_parameters_[0] = ConvertToCppAD(mean);
        model_parameters_[1] = ConvertToCppAD(covariance);

        // Compute the normalization constant based on the updated parameters
        ComputeNormalizationConstant();
    }

    /**
     * @brief Evaluate the normalized multivariate normal PDF.
     *
     * @param x Argument that the PDF is evaluated at.
     * @return Normalized PDF value.
     */
    double EvaluateModelNormalized(const Eigen::VectorXd &x)
    {
        return norm_const_ * EvaluateModel(x);
    }

    /**
     * @brief Evaluate the normalized log multivariate normal PDF.
     *
     * @param x Argument that the log PDF is evaluated at.
     * @return Normalized log PDF value.
     */
    double EvaluateLogModelNormalized(const Eigen::VectorXd &x)
    {
        return std::log(norm_const_) + EvaluateLogModel(x);
    }

    /**
     * @brief Evaluate the gradient of the normalized multivariate normal PDF.
     *
     * @param x Argument that the gradient is evaluated at.
     * @return Normalized PDF gradient.
     */
    Eigen::VectorXd EvaluateModelGradNormalized(const Eigen::VectorXd &x)
    {
        return norm_const_ * EvaluateModelGrad(x);
    }

    /**
     * @brief Get the normalization constant.
     *
     * @return Normalization constant of the multivariate normal distribution.
     */
    double GetNormalizationConstant() { return norm_const_; }

protected:
    /**
     * @brief Compute the normalization constant for the multivariate normal.
     *
     */
    void ComputeNormalizationConstant()
    {
        norm_const_ = 1.0 /
                      (std::pow(2.0 * M_PI, dimension_ / 2.0) * std::sqrt(ConvertFromCppAD(model_parameters_[1]).determinant()));
    }

    double norm_const_; ///< Normalization constant.
};

#endif
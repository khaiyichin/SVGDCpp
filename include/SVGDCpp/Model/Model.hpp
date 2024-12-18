/**
 * @file Model.hpp
 * @author Khai Yi Chin (khaiyichin@gmail.com)
 * @brief Model class header
 *
 * @copyright Copyright (c) 2024 Khai Yi Chin
 *
 */

#ifndef SVGDCPP_MODEL_HPP
#define SVGDCPP_MODEL_HPP

#include "../Core.hpp"

/**
 * @class Model
 * @brief This can be used as is or derived from to define a model.
 * @ingroup Model_Module
 */
class Model
{
public:
    /**
     * @brief Default constructor
     * @details This should almost never be called directly. Use instead @ref Model(const size_t &dim).
     */
    Model() {}

    virtual ~Model() {}

    /**
     * @brief Construct a new Model object by deep copying variables from another Model object.
     *
     * @param obj Model object to be copied.
     */
    Model(const Model &obj)
    {
        *this = obj;
    }

    /**
     * @brief Construct a new Model object.
     * @details This is the preferred method to instantiate a Model class if no derived classed from Model is used.
     *
     * @param dim Dimension of the problem, i.e., dimension of the particle coordinates.
     */
    Model(const size_t &dim) : dimension_(dim) {}

    /**
     * @brief Sum `*this` with @a obj to produce a new @ref Model object.
     *
     * @param obj Another parametrized @ref Model object.
     * @return A new @ref Kernel object whose function is the sum of `*this` and @a obj functions.
     */
    Model operator+(const Model &obj)
    {
        // Ensure that dimensions are correct
        if (dimension_ != obj.dimension_)
        {
            throw DimensionMismatchException("Only models with the same variable dimensions can be added.");
        }

        if (!model_fun_ || !obj.model_fun_)
        {
            throw UnsetException("One of the model functions is unset; functional composition requires both model functions to be set.");
        }

        // Create new object and combine model parameters
        Model new_obj(dimension_);
        new_obj.model_parameters_ = model_parameters_;
        new_obj.model_parameters_.insert(
            new_obj.model_parameters_.end(),
            obj.model_parameters_.begin(),
            obj.model_parameters_.end());

        // Define the sum of two models
        auto sum_model_fun = [this, &obj](const VectorXADd &x, const std::vector<MatrixXADd> &params)
        {
            // Split the parameters
            std::vector<MatrixXADd> params_1(params.begin(), params.begin() + this->model_parameters_.size());
            std::vector<MatrixXADd> params_2(params.begin() + this->model_parameters_.size(), params.end());

            // Compute the result
            VectorXADd result(1);
            result << this->ModelFun(x, params_1).array() + obj.ModelFun(x, params_2).array();
            return result;
        };

        new_obj.UpdateModel(sum_model_fun);

        return new_obj;
    }

    /**
     * @brief Subtract `*this` by @a obj to produce a new @ref Model object.
     *
     * @param obj Another parametrized @ref Model object.
     * @return A new @ref Kernel object whose function is the difference of `*this` and @a obj functions.
     */
    Model operator-(const Model &obj)
    {
        // Ensure that dimensions are correct
        if (dimension_ != obj.dimension_)
        {
            throw DimensionMismatchException("Only models with the same variable dimensions can be added.");
        }

        if (!model_fun_ || !obj.model_fun_)
        {
            throw UnsetException("One of the model functions is unset; functional composition requires both model functions to be set.");
        }

        // Create new object and combine model parameters
        Model new_obj(dimension_);
        new_obj.model_parameters_ = model_parameters_;
        new_obj.model_parameters_.insert(
            new_obj.model_parameters_.end(),
            obj.model_parameters_.begin(),
            obj.model_parameters_.end());

        // Define the sum of two models
        auto sum_model_fun = [this, &obj](const VectorXADd &x, const std::vector<MatrixXADd> &params)
        {
            // Split the parameters
            std::vector<MatrixXADd> params_1(params.begin(), params.begin() + this->model_parameters_.size());
            std::vector<MatrixXADd> params_2(params.begin() + this->model_parameters_.size(), params.end());

            // Compute the result
            VectorXADd result(1);
            result << this->ModelFun(x, params_1).array() - obj.ModelFun(x, params_2).array();
            return result;
        };

        new_obj.UpdateModel(sum_model_fun);

        return new_obj;
    }

    /**
     * @brief Multiply `*this` with @a obj to produce a new @ref Model object.
     *
     * @param obj Another parametrized @ref Model object.
     * @return A new @ref Kernel object whose function is the product of `*this` and @a obj functions.
     */
    Model operator*(const Model &obj)
    {
        // Ensure that dimensions are correct
        if (dimension_ != obj.dimension_)
        {
            throw DimensionMismatchException("Only models with the same variable dimensions can be multiplied.");
        }

        if (!model_fun_ || !obj.model_fun_)
        {
            throw UnsetException("One of the model functions is unset; functional composition requires both model functions to be set.");
        }

        // Create new object and combine model parameters
        Model new_obj(dimension_);
        new_obj.model_parameters_ = model_parameters_;
        new_obj.model_parameters_.insert(
            new_obj.model_parameters_.end(),
            obj.model_parameters_.begin(),
            obj.model_parameters_.end());

        // Define the product of two models
        auto product_model_fun = [this, &obj](const VectorXADd &x, const std::vector<MatrixXADd> &params)
        {
            // Split the parameters
            std::vector<MatrixXADd> params_1(params.begin(), params.begin() + this->model_parameters_.size());
            std::vector<MatrixXADd> params_2(params.begin() + this->model_parameters_.size(), params.end());

            // Compute the result
            VectorXADd result(1);
            result << this->ModelFun(x, params_1).array() * obj.ModelFun(x, params_2).array();
            return result;
        };

        new_obj.UpdateModel(product_model_fun);

        return new_obj;
    }

    /**
     * @brief Divide `*this` by @a obj to produce a new @ref Model object.
     *
     * @param obj Another parametrized @ref Model object.
     * @return A new @ref Kernel object whose function is the quotient of `*this` and @a obj functions.
     */
    Model operator/(const Model &obj)
    {
        // Ensure that dimensions are correct
        if (dimension_ != obj.dimension_)
        {
            throw DimensionMismatchException("Only models with the same variable dimensions can be multiplied.");
        }

        if (!model_fun_ || !obj.model_fun_)
        {
            throw UnsetException("One of the model functions is unset; functional composition requires both model functions to be set.");
        }

        // Create new object and combine model parameters
        Model new_obj(dimension_);
        new_obj.model_parameters_ = model_parameters_;
        new_obj.model_parameters_.insert(
            new_obj.model_parameters_.end(),
            obj.model_parameters_.begin(),
            obj.model_parameters_.end());

        // Define the quotient of two models
        auto quotient_model_fun = [this, &obj](const VectorXADd &x, const std::vector<MatrixXADd> &params)
        {
            // Split the parameters
            std::vector<MatrixXADd> params_1(params.begin(), params.begin() + this->model_parameters_.size());
            std::vector<MatrixXADd> params_2(params.begin() + this->model_parameters_.size(), params.end());

            // Compute the result
            VectorXADd result(1);
            result << this->ModelFun(x, params_1).array() / obj.ModelFun(x, params_2).array();
            return result;
        };

        new_obj.UpdateModel(quotient_model_fun);

        return new_obj;
    }

    /**
     * @brief Assignment operator.
     */
    Model &operator=(const Model &obj)
    {
        dimension_ = obj.dimension_;
        model_parameters_ = obj.model_parameters_;
        model_fun_ = obj.model_fun_;
        model_fun_ad_ = obj.model_fun_ad_;
        logmodel_fun_ad_ = obj.logmodel_fun_ad_;

        return *this;
    }

    /**
     * @brief Copy an instance of this object into a unique pointer.
     *
     * @return Unique pointer to a copy of *this.
     */
    virtual std::unique_ptr<Model> CloneUniquePointer() const
    {
        return std::make_unique<Model>(*this);
    }

    /**
     * @brief Copy an instance of this object into a shared pointer.
     *
     * @return Shared pointer to a copy of *this.
     */
    virtual std::shared_ptr<Model> CloneSharedPointer() const
    {
        return std::make_shared<Model>(*this);
    }

    /**
     * @brief Initialize the model.
     * @details This is called by the @ref SVGD::Initialize method. Internally this sets up the CppAD function.
     * This should be called if the model's function has been updated using @ref UpdateModel.
     */
    virtual void Initialize()
    {
        // Ensure that the dimension has been set properly
        if (dimension_ <= 0)
        {
            throw UnsetException("Model dimension (" + std::to_string(dimension_) + ") is improperly or not set.");
        }

        // Set up the CppAD functions
        SetupADFun();
    }

    /**
     * @brief Evaluate the model.
     * @details Override this function in the derived class if you have and wish to use a closed-form function
     * directly, bypassing the use of @ref Model::ModelFun.
     * @warning Override this in the derived class only if you **do not** intend to compose a new model from said derived class.
     * This is because functional composition relies on the @ref Model::ModelFun method, which means that the resulting model
     * will not use the overridden function.
     * @param x Argument that the model is evaluated at.
     * @return Evaluated model value.
     */
    virtual double EvaluateModel(const Eigen::VectorXd &x)
    {
        return model_fun_ad_.Forward(0, x)(0, 0);
    }

    /**
     * @brief Evaluate the log model.
     * @details Override this function in the derived class if you have and wish to use a closed-form function
     * directly, bypassing the use of @ref Model::ModelFun.
     * @warning Override this in the derived class only if you **do not** intend to compose a new model from said derived class.
     * This is because functional composition relies on the @ref Model::ModelFun method, which means that the resulting model
     * will not use the overridden function.
     * @param x Argument that the log model is evaluated at.
     * @return Evaluated log model value.
     */
    virtual double EvaluateLogModel(const Eigen::VectorXd &x)
    {
        return logmodel_fun_ad_.Forward(0, x)(0, 0);
    }

    /**
     * @brief Evaluate the gradient of the model.
     * @details Override this function in the derived class if you have and wish to use a closed-form function
     * instead of relying on automatic differentiation. This will bypass the use of @ref Model::ModelFun.
     * @warning Override this in the derived class only if you **do not** intend to compose a new model from said derived class.
     * This is because functional composition relies on the @ref Model::ModelFun method, which means that the resulting model
     * will not use the overridden function.
     * @param x Argument that the gradient is evaluated at.
     * @return Evaluated model gradient vector.
     */
    virtual Eigen::VectorXd EvaluateModelGrad(const Eigen::VectorXd &x)
    {
        return model_fun_ad_.Jacobian(x);
    }

    /**
     * @brief Evaluate the gradient of the log model.
     * @details Override this function in the derived class if you have and wish to use a closed-form function
     * instead of relying on automatic differentiation. This will bypass the use of @ref Model::ModelFun.
     * @warning Override this in the derived class only if you **do not** intend to compose a new model from said derived class.
     * This is because functional composition relies on the @ref Model::ModelFun method, which means that the resulting model
     * will not use the overridden function.
     * @param x Argument that the gradient of the log model is evaluated at.
     * @return Evaluated log model gradient vector.
     */
    virtual Eigen::VectorXd EvaluateLogModelGrad(const Eigen::VectorXd &x)
    {
        return logmodel_fun_ad_.Jacobian(x);
    }

    /**
     * @brief Evaluate the Hessian of the model.
     * @details Override this function in the derived class if you have and wish to use a closed-form function
     * instead of relying on automatic differentiation. This will bypass the use of @ref Model::ModelFun.
     * @warning Override this in the derived class only if you **do not** intend to compose a new model from said derived class.
     * This is because functional composition relies on the @ref Model::ModelFun method, which means that the resulting model
     * will not use the overridden function.
     * @param x Argument that the Hessian of the model is evaluated at.
     * @return Evaluated model Hessian matrix.
     */
    virtual Eigen::MatrixXd EvaluateModelHessian(const Eigen::VectorXd &x)
    {
        // Model is a scalar function (1-D output), so evaluating Hessian at function of index 0
        return Eigen::Map<Eigen::Matrix<double, -1, -1>>(model_fun_ad_.Hessian(x, 0).data(), dimension_, dimension_).transpose(); // need to transpose because of column-major (default) storage
    }

    /**
     * @brief Evaluate the Hessian of the log model.
     * @details Override this function in the derived class if you have and wish to use a closed-form function
     * instead of relying on automatic differentiation. This will bypass the use of @ref Model::ModelFun.
     * @warning Override this in the derived class only if you **do not** intend to compose a new model from said derived class.
     * This is because functional composition relies on the @ref Model::ModelFun method, which means that the resulting model
     * will not use the overridden function.
     * @param x Argument that the Hessian of the log model is evaluated at.
     * @return Evaluated log model Hessian matrix.
     */
    virtual Eigen::MatrixXd EvaluateLogModelHessian(const Eigen::VectorXd &x)
    {
        // Model is a scalar function (1-D output), so evaluating Hessian at function of index 0
        return Eigen::Map<Eigen::Matrix<double, -1, -1>>(logmodel_fun_ad_.Hessian(x, 0).data(), dimension_, dimension_).transpose(); // need to transpose because of column-major (default) storage
    }

    /**
     * @brief Update the model parameters.
     *
     * @param params Vector of variable-sized Eigen objects.
     */
    virtual void UpdateParameters(const std::vector<Eigen::MatrixXd> &params)
    {
        std::vector<MatrixXADd> converted_params(params.size());

        for (size_t i = 0; i < params.size(); ++i)
        {
            // Convert into CppAD::AD<double> type
            converted_params[i] = ConvertToCppAD(params[i]);
        }

        model_parameters_ = converted_params;
    }

    /**
     * @brief Get the model parameters.
     *
     * @return Vector of variable-sized objects.
     */
    std::vector<Eigen::MatrixXd> GetParameters() const
    {
        std::vector<Eigen::MatrixXd> converted_params(model_parameters_.size());

        for (size_t i = 0; i < model_parameters_.size(); ++i)
        {
            // Convert to double type
            converted_params[i] = ConvertFromCppAD(model_parameters_[i]);
        }

        return converted_params;
    }

    /**
     * @brief Execute methods required at each step.
     * @details Override this method to include methods that you need to have run every step of the iteration.
     *
     */
    virtual void Step() {}

    /**
     * @brief Update the model symbolic function.
     *
     * @param model_fun STL function defining the model function.
     * The function argument should be a `const &` @ref VectorXADd and a `const &` @ref MatrixXADd STL vector; it returns a @ref VectorXADd.
     */
    void UpdateModel(std::function<VectorXADd(const VectorXADd &, const std::vector<MatrixXADd> &)> model_fun)
    {
        model_fun_ = model_fun;
    }

protected:
    /**
     * @brief Symbolic function of the model, used by CppAD to compute derivatives.
     *
     * @param x Argument that the model is evaluated at (the independent variable).
     * @param params Parameters of the model function.
     * @return Output of the model function (the dependent variable).
     */
    VectorXADd ModelFun(const VectorXADd &x, const std::vector<MatrixXADd> &params) const
    {
        // Ensure that the model function has been set
        if (!model_fun_)
        {
            throw UnsetException("Model function is unset.");
        }

        return model_fun_(x, params);
    }

    /**
     * @brief Symbolic function of the log model, used by CppAD to compute derivatives.
     *
     * @param x Argument that the log model is evaluated at (the independent variable).
     * @return Output of the log model function (the dependent variable).
     */
    virtual VectorXADd LogModelFun(const VectorXADd &x) const
    {
        return ModelFun(x, model_parameters_).array().log();
    }

    int dimension_ = -1; ///< Dimension of the particle coordinates.

    std::vector<MatrixXADd> model_parameters_; ///< Parameters of the model function.

private:
    /**
     * @brief Setup the CppAD function.
     *
     */
    virtual void SetupADFun()
    {
        VectorXADd x_model_ad(dimension_), y_model_ad(dimension_),
            x_logmodel_ad(dimension_), y_logmodel_ad(dimension_);

        // Setup model
        CppAD::Independent(x_model_ad); // start recording sequence

        y_model_ad = ModelFun(x_model_ad, model_parameters_);

        model_fun_ad_.Dependent(x_model_ad, y_model_ad); // store operation sequence and stop recording

        model_fun_ad_.optimize();

        // Setup logmodel
        CppAD::Independent(x_logmodel_ad); // start recording sequence

        y_logmodel_ad = LogModelFun(x_logmodel_ad);

        logmodel_fun_ad_.Dependent(x_logmodel_ad, y_logmodel_ad); // store operation sequence and stop recording

        logmodel_fun_ad_.optimize();
    }

    std::function<VectorXADd(const VectorXADd &, const std::vector<MatrixXADd> &)> model_fun_; ///< Symbolic function of the model.

    CppAD::ADFun<double> model_fun_ad_; ///< CppAD function of the model.

    CppAD::ADFun<double> logmodel_fun_ad_; ///< CppAD function of the log model.
};

#endif
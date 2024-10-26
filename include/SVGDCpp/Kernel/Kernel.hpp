/**
 * @file Kernel.hpp
 * @author Khai Yi Chin (khaiyichin@gmail.com)
 * @brief Kernel class header
 * @version 0.1
 * @date 2024-03-23
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef SVGDCPP_KERNEL_HPP
#define SVGDCPP_KERNEL_HPP

#include "../Core.hpp"

/**
 * @class Kernel
 * @brief This can be used as is or derived from to define a kernel function.
 * @ingroup Kernel_Module
 */
class Kernel
{
public:
    /**
     * @brief Default constructor.
     * @details This should almost never be called directly. Use instead @ref Kernel(const size_t &dim).
     */
    Kernel() {}

    /**
     * @brief Default destructor.
     *
     */
    virtual ~Kernel() {}

    /**
     * @brief Copy constructor.
     */
    Kernel(const Kernel &obj)
    {
        *this = obj;
    }

    /**
     * @brief Construct a new @ref Kernel object.
     *
     * @param dim Dimensions of the particle coordinates.
     */
    Kernel(const size_t &dim) : dimension_(dim), location_vec_ad_(VectorXADd::Zero(dim)) {}

    /**
     * @brief Sum `*this` with @a obj to produce a new @ref Kernel object.
     *
     * @param obj Another parametrized @ref Kernel object.
     * @return A new @ref Kernel object whose function is the sum of `*this` and @a obj functions.
     */
    Kernel operator+(const Kernel &obj)
    {
        // Ensure that dimensions are correct
        if (dimension_ != obj.dimension_)
        {
            throw DimensionMismatchException("Only kernels with the same variable dimensions can be added.");
        }

        if (!kernel_fun_ || !obj.kernel_fun_)
        {
            throw UnsetException("One of the kernel functions is unset; functional composition requires both kernel functions to be set.");
        }

        // Create new object and combine kernel parameters
        Kernel new_obj(this->dimension_);
        new_obj.kernel_parameters_ = kernel_parameters_;
        new_obj.kernel_parameters_.insert(
            new_obj.kernel_parameters_.end(),
            obj.kernel_parameters_.begin(),
            obj.kernel_parameters_.end());

        // Define the sum of two kernels
        auto sum_kernel_fun = [this, &obj](const VectorXADd &x, const std::vector<MatrixXADd> &params, const VectorXADd &location) -> VectorXADd
        {
            // Split the parameters
            std::vector<MatrixXADd> params_1(params.begin(), params.begin() + this->kernel_parameters_.size());
            std::vector<MatrixXADd> params_2(params.begin() + this->kernel_parameters_.size(), params.end());

            VectorXADd result(1);
            result << this->KernelFun(x, params_1, location).array() + obj.KernelFun(x, params_2, location).array();
            return result;
        };

        new_obj.UpdateKernel(sum_kernel_fun);

        return new_obj;
    }

    /**
     * @brief Subtract `*this` by @a obj to produce a new @ref Kernel object.
     *
     * @param obj Another parametrized @ref Kernel object.
     * @return A new @ref Kernel object whose function is the difference of `*this` and @a obj functions.
     */
    Kernel operator-(const Kernel &obj)
    {
        // Ensure that dimensions are correct
        if (dimension_ != obj.dimension_)
        {
            throw DimensionMismatchException("Only kernels with the same variable dimensions can be added.");
        }

        if (!kernel_fun_ || !obj.kernel_fun_)
        {
            throw UnsetException("One of the kernel functions is unset; functional composition requires both kernel functions to be set.");
        }

        // Create new object and combine kernel parameters
        Kernel new_obj(this->dimension_);
        new_obj.kernel_parameters_ = kernel_parameters_;
        new_obj.kernel_parameters_.insert(
            new_obj.kernel_parameters_.end(),
            obj.kernel_parameters_.begin(),
            obj.kernel_parameters_.end());

        // Define the difference of two kernels
        auto difference_kernel_fun = [this, &obj](const VectorXADd &x, const std::vector<MatrixXADd> &params, const VectorXADd &location) -> VectorXADd
        {
            // Split the parameters
            std::vector<MatrixXADd> params_1(params.begin(), params.begin() + this->kernel_parameters_.size());
            std::vector<MatrixXADd> params_2(params.begin() + this->kernel_parameters_.size(), params.end());

            VectorXADd result(1);
            result << this->KernelFun(x, params_1, location).array() - obj.KernelFun(x, params_2, location).array();
            return result;
        };

        new_obj.UpdateKernel(difference_kernel_fun);

        return new_obj;
    }

    /**
     * @brief Multiply `*this` with @a obj to produce a new @ref Kernel object.
     *
     * @param obj Another parametrized @ref Kernel object.
     * @return A new @ref Kernel object whose function is the product of `*this` and @a obj functions.
     */
    Kernel operator*(const Kernel &obj)
    {
        // Ensure that dimensions are correct
        if (dimension_ != obj.dimension_)
        {
            throw DimensionMismatchException("Only kernels with the same variable dimensions can be multiplied.");
        }

        if (!kernel_fun_ || !obj.kernel_fun_)
        {
            throw UnsetException("One of the kernel functions is unset; functional composition requires both kernel functions to be set.");
        }

        // Define new object and combine kernel parameters
        Kernel new_obj(dimension_);
        new_obj.kernel_parameters_ = kernel_parameters_;
        new_obj.kernel_parameters_.insert(
            new_obj.kernel_parameters_.end(),
            obj.kernel_parameters_.begin(),
            obj.kernel_parameters_.end());

        // Define the quotient of two kernels
        auto product_kernel_fun = [this, &obj](const VectorXADd &x, const std::vector<MatrixXADd> &params, const VectorXADd &location) -> VectorXADd
        {
            // Split the parameters
            std::vector<MatrixXADd> params_1(params.begin(), params.begin() + this->kernel_parameters_.size());
            std::vector<MatrixXADd> params_2(params.begin() + this->kernel_parameters_.size(), params.end());

            VectorXADd result(1);
            result << this->KernelFun(x, params_1, location).array() * obj.KernelFun(x, params_2, location).array();
            return result;
        };

        new_obj.UpdateKernel(product_kernel_fun);

        return new_obj;
    }

    /**
     * @brief Multiply `*this` by @a obj to produce a new @ref Kernel object.
     *
     * @param obj Another parametrized @ref Kernel object.
     * @return A new @ref Kernel object whose function is the quotient of `*this` and @a obj functions.
     */
    Kernel operator/(const Kernel &obj)
    {
        // Ensure that dimensions are correct
        if (dimension_ != obj.dimension_)
        {
            throw DimensionMismatchException("Only kernels with the same variable dimensions can be multiplied.");
        }

        if (!kernel_fun_ || !obj.kernel_fun_)
        {
            throw UnsetException("One of the kernel functions is unset; functional composition requires both kernel functions to be set.");
        }

        // Define new object and combine kernel parameters
        Kernel new_obj(dimension_);
        new_obj.kernel_parameters_ = kernel_parameters_;
        new_obj.kernel_parameters_.insert(
            new_obj.kernel_parameters_.end(),
            obj.kernel_parameters_.begin(),
            obj.kernel_parameters_.end());

        // Define the quotient of two kernels
        auto quotient_kernel_fun = [this, &obj](const VectorXADd &x, const std::vector<MatrixXADd> &params, const VectorXADd &location) -> VectorXADd
        {
            // Split the parameters
            std::vector<MatrixXADd> params_1(params.begin(), params.begin() + this->kernel_parameters_.size());
            std::vector<MatrixXADd> params_2(params.begin() + this->kernel_parameters_.size(), params.end());

            VectorXADd result(1);
            result << this->KernelFun(x, params_1, location).array() / obj.KernelFun(x, params_2, location).array();
            return result;
        };

        new_obj.UpdateKernel(quotient_kernel_fun);

        return new_obj;
    }

    /**
     * @brief Assignment operator.
     */
    Kernel &operator=(const Kernel &obj)
    {
        dimension_ = obj.dimension_;
        location_vec_ad_ = obj.location_vec_ad_;
        kernel_parameters_ = obj.kernel_parameters_;
        kernel_fun_ = obj.kernel_fun_;
        kernel_fun_ad_ = CppAD::ADFun<double>(); // not copying over ADFun object to prevent complications when using in multi-threaded mode

        return *this;
    }

    /**
     * @brief Copy an instance of this object into a unique pointer.
     *
     * @return Unique pointer to a copy of *this.
     */
    virtual std::unique_ptr<Kernel> CloneUniquePointer() const
    {
        return std::make_unique<Kernel>(*this);
    }

    /**
     * @brief Copy an instance of this object into a shared pointer.
     *
     * @return Shared pointer to a copy of *this.
     */
    virtual std::shared_ptr<Kernel> CloneSharedPointer() const
    {
        return std::make_shared<Kernel>(*this);
    }

    /**
     * @brief Initialize the kernel.
     * @details This is called by the @ref SVGD::Initialize method. Internally this sets up the CppAD function.
     * This should be called if the kernel's function has been updated using @ref UpdateKernel.
     */
    virtual void Initialize()
    {
        SetupADFun();
    }

    /**
     * @brief Evaluate the kernel function.
     * @details Override this function in the derived class if you have and wish to use a closed-form function
     * directly, bypassing the use of @ref Kernel::KernelFun.
     * @warning Override this in the derived class only if you **do not** intend to compose a new kernel from said derived class.
     * This is because functional composition relies on the @ref Kernel::KernelFun method, which means that the resulting kernel
     * will not use the overridden function.
     * @param x Argument that the kernel is evaluated at.
     * @return Evaluated kernel function value.
     */
    virtual double EvaluateKernel(const Eigen::VectorXd &x)
    {
        return kernel_fun_ad_.Forward(0, x)(0, 0);
    }

    /**
     * @brief Evaluate the gradient of the kernel function.
     * @details Override this function in the derived class if you have and wish to use a closed-form function
     * instead of relying on automatic differentiation. This will bypass the use of @ref Kernel::KernelFun.
     * @warning Override this in the derived class only if you **do not** intend to compose a new kernel from said derived class.
     * This is because functional composition relies on the @ref Kernel::KernelFun method, which means that the resulting kernel
     * will not use the overridden function.
     * @param x Argument that the gradient is evaluated at.
     * @return Evaluated gradient vector.
     */
    virtual Eigen::VectorXd EvaluateKernelGrad(const Eigen::VectorXd &x)
    {
        return kernel_fun_ad_.Jacobian(x);
    }

    /**
     * @brief Update the kernel parameters.
     *
     * @param params Vector of variable-sized Eigen objects.
     */
    virtual void UpdateParameters(const std::vector<Eigen::MatrixXd> &params)
    {
        std::vector<MatrixXADd> converted_params(params.size());

        // Verify that the parameters all have # rows == dimension_
        for (size_t i = 0; i < params.size(); ++i)
        {
            if (params[i].rows() != dimension_)
            {
                throw DimensionMismatchException("Dimension mismatch between provided parameters and model dimension (" + std::to_string(params[i].rows()) + " vs. " + std::to_string(dimension_) + ").");
            }

            // Convert into CppAD::AD<double> type
            converted_params[i] = ConvertToCppAD(params[i]);
        }

        kernel_parameters_ = converted_params;
    };

    /**
     * @brief Update the reference location (i.e., 2nd argument in k(x, x')) at which the kernel is computed with respect to.
     *
     * @param x Variable-sized Eigen vector.
     */
    virtual void UpdateLocation(const Eigen::VectorXd &x)
    {
        location_vec_ad_ = ConvertToCppAD(x);
    }

    /**
     * @brief Get the kernel parameters.
     *
     * @return Vector of variable-sized objects.
     */
    std::vector<Eigen::MatrixXd> GetParameters() const
    {
        std::vector<Eigen::MatrixXd> converted_params(kernel_parameters_.size());

        for (size_t i = 0; i < kernel_parameters_.size(); ++i)
        {
            // Convert to double type
            converted_params[i] = ConvertFromCppAD(kernel_parameters_[i]);
        }

        return converted_params;
    }

    /**
     * @brief Execute methods required for each step.
     * @details Override this method to include methods that you need to have run every step of the iteration
     * @a e.g., computing the scale parameter of the kernel function.
     *
     */
    virtual void Step() {}

    /**
     * @brief Update the kernel symbolic function.
     *
     * @param kernel_fun STL function defining the kernel function.
     * The function argument should be a `const &` @ref VectorXADd, a `const &` @ref MatrixXADd STL vector, and a `const &` @ref VectorXADd; it returns a @ref VectorXADd.
     */
    void UpdateKernel(std::function<VectorXADd(const VectorXADd &, const std::vector<MatrixXADd> &, const VectorXADd &)> kernel_fun)
    {
        kernel_fun_ = kernel_fun;
    }

protected:
    /**
     * @brief Symbolic function of the kernel, used by CppAD to compute derivatives.
     *
     * @param x Argument that the kernel is evaluated at (i.e., the 1st argument in k(x, x'): the independent variable to be differentiated).
     * @param params Parameters of the kernel function.
     * @param location Reference location at which the kernel is evaluated against (i.e., the 2nd argument in k(x, x'): the kernel's origin parameter).
     * @return Output of the kernel function (the dependent variable).
     */
    virtual VectorXADd KernelFun(const VectorXADd &x, const std::vector<MatrixXADd> &params, const VectorXADd &location) const
    {
        // Ensure that the kernel function has been set
        if (!kernel_fun_)
        {
            throw UnsetException("Kernel function is unset.");
        }

        return kernel_fun_(x, params, location);
    }

    int dimension_ = -1; ///< Dimension of the particle coordinates.

    VectorXADd location_vec_ad_; ///< Location at which the kernel is evaluated with respect to.

    std::vector<MatrixXADd> kernel_parameters_; ///< Parameters of the kernel function.

private:
    /**
     * @brief Setup the CppAD function.
     *
     */
    virtual void SetupADFun()
    {
        VectorXADd x_kernel_ad(dimension_), y_kernel_ad(dimension_);

        // Setup kernel
        CppAD::Independent(x_kernel_ad); // start recording sequence

        y_kernel_ad = KernelFun(x_kernel_ad, kernel_parameters_, location_vec_ad_);

        kernel_fun_ad_.Dependent(x_kernel_ad, y_kernel_ad); // store operation sequence and stop recording

        /*
            Choosing not to optimize because the function re-recording happens at every iteration, which is expensive
        */
        // kernel_fun_ad_.optimize();
    }

    std::function<VectorXADd(const VectorXADd &, const std::vector<MatrixXADd> &, const VectorXADd &)> kernel_fun_; ///< Symbolic function of the kernel.

    CppAD::ADFun<double> kernel_fun_ad_; ///< CppAD function of the kernel.
};

#endif
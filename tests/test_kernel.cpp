#include "doctest.h"
#include "Core"

TEST_CASE("Kernel Class")
{
    int low_dim = 2, high_dim = 5;

    Kernel k_none;
    Kernel k_low_dim(low_dim);
    Kernel k_high_dim(high_dim);

    // Define a vector parameter
    Eigen::VectorXd param_1(high_dim);
    param_1 << 5.0, 0.7, 3.0, 0.011, 1.3;

    // Define test locations
    Eigen::VectorXd loc_low_dim(low_dim);
    loc_low_dim << 10.9, -0.2;

    Eigen::VectorXd loc_high_dim(high_dim);
    loc_high_dim << 0.5519, 34.08, 0.26, -6.922, 0.94;

    // Define sum of absolute difference kernel
    auto sum_of_diff_fun =
        [](const VectorXADd &x, const std::vector<MatrixXADd> &params, const VectorXADd &location) -> VectorXADd // dummy params
    {
        VectorXADd result(1);
        result << (x - location).sum();
        return result;
    };

    // Define polynomial kernel
    auto polynomial_fun =
        [](const VectorXADd &x, const std::vector<MatrixXADd> &params, const VectorXADd &location) -> VectorXADd
    {
        VectorXADd result(1), param = params[0]; // implicitly convert params[0] into a vector
        result << CppAD::pow(1.0 + (param.array() * x.array()).matrix().dot(location), 2.0);
        return result;
    };

    // Define test inputs
    Eigen::Vector2d input_vec_low_dim(17.0, 19.0);
    Eigen::VectorXd input_vec_high_dim(high_dim);
    input_vec_high_dim << 0.2003, 2.9, 31, 3.07, 0.43;

    SUBCASE("Constructors")
    {
        SUBCASE("Default constructor")
        {
            CHECK(k_none.GetParameters().empty());
            CHECK(k_low_dim.GetParameters().empty());
        }

        SUBCASE("Copy constructor")
        {
            // Set kernel function
            k_low_dim.UpdateKernel(sum_of_diff_fun);
            k_low_dim.UpdateLocation(loc_low_dim);
            k_low_dim.Initialize();

            // Create a copy
            Kernel k_low_dim_copied(k_low_dim);

            CHECK(k_low_dim.EvaluateKernel(input_vec_low_dim) == doctest::Approx(k_low_dim_copied.EvaluateKernel(input_vec_low_dim)));
        }
    }

    // Set kernel function
    k_low_dim.UpdateKernel(sum_of_diff_fun);
    k_low_dim.UpdateLocation(loc_low_dim);
    k_low_dim.Initialize();

    k_high_dim.UpdateKernel(polynomial_fun);
    k_high_dim.UpdateParameters({param_1});
    k_high_dim.UpdateLocation(loc_high_dim);
    k_high_dim.Initialize();

    SUBCASE("Operators")
    {
        SUBCASE("Kernel composition operators")
        {
            // Check error handling with kernel dimension mismatch
            CHECK_THROWS_AS(k_low_dim + k_high_dim, DimensionMismatchException);

            Kernel k_low_dim_another(low_dim);
            k_low_dim_another.UpdateKernel(
                [](const VectorXADd &x, const std::vector<MatrixXADd> &params, const VectorXADd &location)
                {
                    VectorXADd result(1);
                    result << 3.0 * (x - location).sum();
                    return result;
                });

            Kernel k_high_dim_another(high_dim);
            k_high_dim_another.UpdateKernel(
                [](const VectorXADd &x, const std::vector<MatrixXADd> &params, const VectorXADd &location)
                {
                    VectorXADd result(1);
                    result << 2.0 * (1.0 + x.dot(location));
                    return result;
                });

            // Create composed kernels
            Kernel sum_kernel = k_low_dim + k_low_dim_another;
            Kernel difference_kernel = k_high_dim - k_high_dim_another;
            Kernel product_kernel = k_low_dim * k_low_dim_another;
            Kernel quotient_kernel = k_high_dim / k_high_dim_another;

            // Compute and compare expected output (without updating the location)
            double sum_kernel_output = std::abs((input_vec_low_dim).sum()) + (3.0 * (input_vec_low_dim).sum());
            double difference_kernel_output = 1.0 - 2.0;
            double product_kernel_output = std::abs((input_vec_low_dim).sum()) * (3.0 * (input_vec_low_dim).sum());
            double quotient_kernel_output = 1.0 / 2.0;

            CHECK(sum_kernel.EvaluateKernel(input_vec_low_dim) == doctest::Approx(sum_kernel_output));
            CHECK(difference_kernel.EvaluateKernel(input_vec_high_dim) == doctest::Approx(difference_kernel_output));
            CHECK(product_kernel.EvaluateKernel(input_vec_low_dim) == doctest::Approx(product_kernel_output));
            CHECK(quotient_kernel.EvaluateKernel(input_vec_high_dim) == doctest::Approx(quotient_kernel_output));

            // Compute and compare expected output (after updating the location)
            sum_kernel.UpdateLocation(loc_low_dim.normalized());
            difference_kernel.UpdateLocation(loc_high_dim.normalized());
            product_kernel.UpdateLocation(loc_low_dim.normalized());
            quotient_kernel.UpdateLocation(loc_high_dim.normalized());

            sum_kernel.Initialize();
            difference_kernel.Initialize();
            product_kernel.Initialize();
            quotient_kernel.Initialize();

            sum_kernel_output = std::abs((input_vec_low_dim - loc_low_dim.normalized()).sum()) + (3.0 * (input_vec_low_dim - loc_low_dim.normalized()).sum());
            difference_kernel_output = std::pow(1.0 + (param_1.array() * input_vec_high_dim.array()).matrix().dot(loc_high_dim.normalized()), 2.0) - (2.0 * (1.0 + input_vec_high_dim.dot(loc_high_dim.normalized())));
            product_kernel_output = std::abs((input_vec_low_dim - loc_low_dim.normalized()).sum()) * (3.0 * (input_vec_low_dim - loc_low_dim.normalized()).sum());
            quotient_kernel_output = std::pow(1.0 + (param_1.array() * input_vec_high_dim.array()).matrix().dot(loc_high_dim.normalized()), 2.0) / (2.0 * (1.0 + input_vec_high_dim.dot(loc_high_dim.normalized())));

            CHECK(sum_kernel.EvaluateKernel(input_vec_low_dim) == doctest::Approx(sum_kernel_output));
            CHECK(difference_kernel.EvaluateKernel(input_vec_high_dim) == doctest::Approx(difference_kernel_output));
            CHECK(product_kernel.EvaluateKernel(input_vec_low_dim) == doctest::Approx(product_kernel_output));
            CHECK(quotient_kernel.EvaluateKernel(input_vec_high_dim) == doctest::Approx(quotient_kernel_output));
        }

        SUBCASE("Assignment operator")
        {
            Kernel k_low_dim_assigned;
            k_low_dim_assigned = k_low_dim;

            CHECK(k_low_dim.EvaluateKernel(input_vec_low_dim) == doctest::Approx(k_low_dim_assigned.EvaluateKernel(input_vec_low_dim)));
        }
    }

    SUBCASE("Evaluate* functions")
    {
        double output_low_dim = std::abs((input_vec_low_dim - loc_low_dim).sum());
        double output_high_dim = std::pow(1.0 + (param_1.array() * input_vec_high_dim.array()).matrix().dot(loc_high_dim), 2.0);

        Eigen::VectorXd output_low_dim_grad = Eigen::VectorXd::Ones(low_dim);
        Eigen::VectorXd output_high_dim_grad = param_1.array() * loc_high_dim.array();
        double factor = (param_1.array() * loc_high_dim.array() * input_vec_high_dim.array()).sum() + 1.0;
        output_high_dim_grad *= 2.0 * factor;

        SUBCASE("EvaluateKernel function")
        {
            CHECK(k_low_dim.EvaluateKernel(input_vec_low_dim) == doctest::Approx(output_low_dim));
            CHECK(k_high_dim.EvaluateKernel(input_vec_high_dim) == doctest::Approx(output_high_dim));
        }

        SUBCASE("EvaluateKernelGrad function")
        {
            CHECK(k_low_dim.EvaluateKernelGrad(input_vec_low_dim).isApprox(output_low_dim_grad));
            CHECK(k_high_dim.EvaluateKernelGrad(input_vec_high_dim).isApprox(output_high_dim_grad));
        }
    }

    SUBCASE("Parameter, kernel, and location update")
    {
        // Check parameters
        CHECK(k_high_dim.GetParameters().size() == 1);
        CHECK(k_high_dim.GetParameters()[0].isApprox(param_1));

        Kernel k_high_dim_another(high_dim);
        k_high_dim_another.UpdateKernel(sum_of_diff_fun);
        k_high_dim_another.UpdateLocation(loc_high_dim);
        k_high_dim_another.Initialize();

        // Define test input
        Eigen::VectorXd in(high_dim);
        in << -12.441, 3.905, 0.3184, 0.0443, -199.232;

        // Create a sum kernel
        Kernel k_composed_sum(high_dim);
        k_composed_sum = k_high_dim_another + k_high_dim;
        k_composed_sum.UpdateLocation(loc_high_dim);
        k_composed_sum.Initialize();

        // Compute expected outputs for the sum model
        double out_sum_of_abs_diff = ConvertFromCppAD(sum_of_diff_fun(ConvertToCppAD(in), {}, ConvertToCppAD(loc_high_dim)))(0, 0);
        double out_polynomial = ConvertFromCppAD(polynomial_fun(ConvertToCppAD(in), {ConvertToCppAD(param_1)}, ConvertToCppAD(loc_high_dim)))(0, 0);

        Eigen::VectorXd out_sum_of_abs_diff_grad = Eigen::VectorXd::Ones(high_dim);
        Eigen::VectorXd out_polynomial_grad = param_1.array() * loc_high_dim.array();
        double factor = (param_1.array() * loc_high_dim.array() * in.array()).sum() + 1.0;
        out_polynomial_grad *= 2.0 * factor;

        // Check evaluation correctness
        CHECK(k_composed_sum.EvaluateKernel(in) == doctest::Approx(out_sum_of_abs_diff + out_polynomial));
        CHECK(k_composed_sum.EvaluateKernelGrad(in).isApprox(out_sum_of_abs_diff_grad + out_polynomial_grad));

        // Update new location
        Eigen::VectorXd in_new_1(high_dim);
        in_new_1 << 99.3207, -2.0205, 0.2922, 45.001, 6.43;
        k_composed_sum.UpdateLocation(loc_high_dim.normalized());
        k_composed_sum.Initialize();

        // Compute expected outputs for updated (location) sum kernel
        out_sum_of_abs_diff = ConvertFromCppAD(sum_of_diff_fun(ConvertToCppAD(in_new_1), {}, ConvertToCppAD(loc_high_dim.normalized())))(0, 0);
        out_polynomial = ConvertFromCppAD(polynomial_fun(ConvertToCppAD(in_new_1), {ConvertToCppAD(param_1)}, ConvertToCppAD(loc_high_dim.normalized())))(0, 0);

        out_polynomial_grad = param_1.array() * loc_high_dim.normalized().array();
        factor = (param_1.array() * loc_high_dim.normalized().array() * in_new_1.array()).sum() + 1.0;
        out_polynomial_grad *= 2.0 * factor;

        // Re-check evaluation correctness (for updated location)
        CHECK(k_composed_sum.EvaluateKernel(in_new_1) == doctest::Approx(out_sum_of_abs_diff + out_polynomial));
        CHECK(k_composed_sum.EvaluateKernelGrad(in_new_1).isApprox(out_sum_of_abs_diff_grad + out_polynomial_grad));

        // Update new parameters
        Eigen::VectorXd in_new_2(high_dim);
        in_new_2 << -29.55, -43.1, 3.748, 0.2665, 49.002;
        Eigen::VectorXd p1_new_2(high_dim); // skipping p1_new_1 for consistent naming
        p1_new_2 << 9.4, 901.03, -2.348, 3.116, 0.4;

        k_composed_sum.UpdateParameters({p1_new_2});
        k_composed_sum.Initialize();

        // Compute expected outputs for the updated (parameters) sum kernel
        out_sum_of_abs_diff = ConvertFromCppAD(sum_of_diff_fun(ConvertToCppAD(in_new_2), {}, ConvertToCppAD(loc_high_dim.normalized())))(0, 0);
        out_polynomial = ConvertFromCppAD(polynomial_fun(ConvertToCppAD(in_new_2), {ConvertToCppAD(p1_new_2)}, ConvertToCppAD(loc_high_dim.normalized())))(0, 0);

        out_polynomial_grad = p1_new_2.array() * loc_high_dim.normalized().array();
        factor = (p1_new_2.array() * loc_high_dim.normalized().array() * in_new_2.array()).sum() + 1.0;
        out_polynomial_grad *= 2.0 * factor;

        // Re-check evaluation correctness (for updated parameters)
        CHECK(k_composed_sum.EvaluateKernel(in_new_2) == doctest::Approx(out_sum_of_abs_diff + out_polynomial));
        CHECK(k_composed_sum.EvaluateKernelGrad(in_new_2).isApprox(out_sum_of_abs_diff_grad + out_polynomial_grad));
    }
}
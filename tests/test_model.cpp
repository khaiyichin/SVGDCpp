#include "doctest.h"
#include "Core"

TEST_CASE("Model Class")
{
    int low_dim = 2, high_dim = 5;

    Model m_none;
    Model m_low_dim(low_dim);
    Model m_high_dim(high_dim);

    // Define a vector parameter
    Eigen::MatrixXd param_1(low_dim, 1);
    param_1 << 5.0, 0.7;

    // Define a matrix parameter
    Eigen::MatrixXd param_2(low_dim, low_dim);
    param_2 << 0.01, 3.0, 0.011, 1.3;

    // Define 2 parameters for the model
    std::vector<Eigen::MatrixXd> parameters(2);

    parameters[0] = param_1;
    parameters[1] = param_2;

    // Define sum of components function and its double-typed derivatives (only the first function is used by `UpdateModel`)
    // Also the double-typed functions take in only a singular Eigen::MatrixXd for simplicity
    auto component_sum_fun =
        [](const VectorXADd &x, const std::vector<MatrixXADd> &params) -> VectorXADd // dummy params
    {
        VectorXADd result(1);
        result << x.sum();
        return result;
    };

    auto component_sum_fun_grad =
        [](const Eigen::VectorXd &x, const Eigen::MatrixXd &param) -> Eigen::VectorXd // dummy param
    {
        return Eigen::MatrixXd::Ones(x.size(), 1);
    };

    auto component_sum_fun_log_grad =
        [](const Eigen::VectorXd &x, const Eigen::MatrixXd &param) -> Eigen::VectorXd // dummy param
    {
        Eigen::VectorXd result = Eigen::MatrixXd::Ones(x.size(), 1);
        return result / x.sum();
    };

    auto component_sum_fun_hess =
        [](const Eigen::VectorXd &x, const Eigen::MatrixXd &param) -> Eigen::MatrixXd // dummy param
    {
        return Eigen::MatrixXd::Zero(x.size(), x.size());
    };

    auto component_sum_fun_log_hess =
        [](const Eigen::VectorXd &x, const Eigen::MatrixXd &param) -> Eigen::MatrixXd // dummy param
    {
        return -Eigen::MatrixXd::Ones(x.size(), x.size()) / std::pow(x.sum(), 2);
    };

    // Define linear function and its double-typed derivatives (only the first function is used by `UpdateModel`)
    // Also the double-typed functions take in only a singular Eigen::MatrixXd for simplicity
    auto linear_fun =
        [](const VectorXADd &x, const std::vector<MatrixXADd> &params) -> VectorXADd
    {
        VectorXADd result(1);
        result << (params[0].array() * x.array()).sum();
        return result;
    };

    auto linear_fun_grad =
        [](const Eigen::VectorXd &x, const Eigen::MatrixXd &param) -> Eigen::VectorXd
    {
        Eigen::VectorXd result = Eigen::MatrixXd::Zero(param.size(), 1);
        for (int i = 0; i < param.size(); ++i)
        {
            result(i) = param(i);
        }
        return result;
    };

    auto linear_fun_log_grad =
        [linear_fun_grad](const Eigen::VectorXd &x, const Eigen::MatrixXd &param) -> Eigen::VectorXd
    {
        return linear_fun_grad(x, param) / (param.array() * x.array()).sum();
    };

    auto linear_fun_hess =
        [](const Eigen::VectorXd &x, const Eigen::MatrixXd &param) -> Eigen::MatrixXd
    {
        return Eigen::MatrixXd::Zero(x.size(), x.size());
    };

    auto linear_fun_log_hess =
        [](const Eigen::VectorXd &x, const Eigen::MatrixXd &param) -> Eigen::MatrixXd
    {
        Eigen::MatrixXd result(2, 2); // hardcoded solution; hard to generalize to n-dimensions
        result << param(0) * param(0), param(1) * param(0), param(0) * param(1), param(1) * param(1);
        return -result / std::pow((param.array() * x.array()).sum(), 2);
    };

    // Define squared function and its double-typed derivatives (only the first function is used by `UpdateModel`)
    // Also the double-typed functions take in only a singular Eigen::MatrixXd for simplicity
    auto squared_fun =
        [](const VectorXADd &x, const std::vector<MatrixXADd> &params) -> VectorXADd
    {
        VectorXADd result(1);
        result << x.transpose() * params[1] * x;
        return result;
    };

    auto squared_fun_grad =
        [](const Eigen::VectorXd &x, const Eigen::MatrixXd &param) -> Eigen::VectorXd
    {
        return Eigen::Vector2d(2.0 * param(0, 0) * x(0) + (param(0, 1) + param(1, 0)) * x(1),
                               2.0 * param(1, 1) * x(1) + (param(0, 1) + param(1, 0)) * x(0)); // hardcoded solution; hard to generalize to n-dimensions
    };

    auto squared_fun_log_grad =
        [squared_fun_grad](const Eigen::VectorXd &x, const Eigen::MatrixXd &param) -> Eigen::VectorXd
    {
        return squared_fun_grad(x, param) / (x.transpose() * param * x);
    };

    auto squared_fun_hess =
        [](const Eigen::VectorXd &x, const Eigen::MatrixXd &param) -> Eigen::MatrixXd
    {
        Eigen::Matrix2d result; // hardcoded solution; hard to generalize to n-dimensions
        result << 2.0 * param(0, 0), param(0, 1) + param(1, 0), param(0, 1) + param(1, 0), 2.0 * param(1, 1);
        return result;
    };

    auto squared_fun_log_hess =
        [squared_fun_grad, squared_fun_hess](const Eigen::VectorXd &x, const Eigen::MatrixXd &param) -> Eigen::MatrixXd
    {
        double denom = x.transpose() * param * x;
        Eigen::Vector2d grad_val = squared_fun_grad(x, param); // hardcoded solution; hard to generalize to n-dimensions
        Eigen::Matrix2d hess_val = squared_fun_hess(x, param);
        Eigen::Matrix2d result;
        result << -std::pow(grad_val(0), 2),
            -(grad_val(0) * grad_val(1)),
            -(grad_val(0) * grad_val(1)),
            -std::pow(grad_val(1), 2);
        result /= std::pow(denom, 2);
        result += hess_val / denom;
        return result;
    };

    // Define test inputs
    Eigen::Vector2d input_vec_low_dim(17.0, 19.0);
    Eigen::MatrixXd input_vec_high_dim(high_dim, 1);
    input_vec_high_dim << 0.2003, 2.9, 31, 3.07, 0.43;

    SUBCASE("Constructors")
    {
        SUBCASE("Default constructor")
        {
            CHECK(m_none.GetParameters().empty());
            CHECK(m_low_dim.GetParameters().empty());
        }

        SUBCASE("Copy constructor")
        {
            // Set model function
            m_low_dim.UpdateModel(linear_fun);
            m_low_dim.UpdateParameters(parameters);
            m_low_dim.Initialize();

            // Create a copy
            Model m_low_dim_copied(m_low_dim);

            CHECK(m_low_dim.EvaluateModel(input_vec_low_dim) == doctest::Approx(m_low_dim_copied.EvaluateModel(input_vec_low_dim)));
        }
    }

    // Set model function
    m_low_dim.UpdateModel(linear_fun);
    m_low_dim.UpdateParameters(parameters);
    m_low_dim.Initialize();

    m_high_dim.UpdateModel(component_sum_fun);
    m_high_dim.Initialize();

    SUBCASE("Operators")
    {
        SUBCASE("Model composition operators")
        {
            // Check error handling with model dimension mismatch
            CHECK_THROWS_AS(m_low_dim + m_high_dim, DimensionMismatchException);
            CHECK_THROWS_AS(m_low_dim * m_high_dim, DimensionMismatchException);

            Model m_low_dim_another(low_dim);
            m_low_dim_another.UpdateModel(squared_fun);
            m_low_dim_another.UpdateParameters(parameters);

            Model m_high_dim_another(high_dim);
            m_high_dim_another.UpdateModel(
                [](const VectorXADd &x, const std::vector<MatrixXADd> &params)
                {
                    VectorXADd result(1);
                    result << 2 * x.sum();
                    return result;
                });

            Model sum_model_1 = m_low_dim + m_low_dim;
            Model sum_model_2 = m_low_dim + m_low_dim_another;
            Model difference_model = m_low_dim_another - m_low_dim;
            Model product_model = m_high_dim * m_high_dim;
            Model quotient_model = m_high_dim / m_high_dim_another;

            // Initialize models
            sum_model_1.Initialize();
            sum_model_2.Initialize();
            difference_model.Initialize();
            product_model.Initialize();
            quotient_model.Initialize();

            // Compute and compare expected output
            double sum_model_output_1 = 2 * (parameters[0].array() * input_vec_low_dim.array()).sum();
            double sum_model_output_2 = (parameters[0].array() * input_vec_low_dim.array()).sum() +
                                        (input_vec_low_dim.transpose() * parameters[1] * input_vec_low_dim);
            double difference_model_output = (input_vec_low_dim.transpose() * parameters[1] * input_vec_low_dim) -
                                             (parameters[0].array() * input_vec_low_dim.array()).sum();
            double product_model_output = std::pow(input_vec_high_dim.sum(), 2);
            double quotient_model_output = 1.0 / 2.0;

            CHECK(sum_model_1.EvaluateModel(input_vec_low_dim) == doctest::Approx(sum_model_output_1));
            CHECK(sum_model_2.EvaluateModel(input_vec_low_dim) == doctest::Approx(sum_model_output_2));
            CHECK(difference_model.EvaluateModel(input_vec_low_dim) == doctest::Approx(difference_model_output));
            CHECK(product_model.EvaluateModel(input_vec_high_dim) == doctest::Approx(product_model_output));
            CHECK(quotient_model.EvaluateModel(input_vec_high_dim) == doctest::Approx(quotient_model_output));
        }

        SUBCASE("Assignment operator")
        {
            // Assign a copy
            Model m_low_dim_assigned;
            m_low_dim_assigned = m_low_dim;

            CHECK(m_low_dim.EvaluateModel(input_vec_low_dim) == doctest::Approx(m_low_dim_assigned.EvaluateModel(input_vec_low_dim)));
        }
    }

    SUBCASE("Evaluate* functions")
    {
        // Define a square function model
        Model m_squared_low_dim(2);
        m_squared_low_dim.UpdateModel(squared_fun);
        m_squared_low_dim.UpdateParameters(parameters);
        m_squared_low_dim.Initialize();

        std::vector<MatrixXADd> converted_parameters(parameters.size());

        for (size_t i = 0; i < parameters.size(); ++i)
        {
            converted_parameters[i] = ConvertToCppAD(parameters[i]);
        }

        // Compute expected function output
        VectorXADd input_vec_low_dim_converted = ConvertToCppAD(input_vec_low_dim);
        VectorXADd input_vec_high_dim_converted = ConvertToCppAD(input_vec_high_dim);

        double output_low_dim = ConvertFromCppAD(linear_fun(input_vec_low_dim_converted, converted_parameters))(0, 0);
        double output_squared_low_dim = ConvertFromCppAD(squared_fun(input_vec_low_dim_converted, converted_parameters))(0, 0);
        double output_high_dim = ConvertFromCppAD(component_sum_fun(input_vec_high_dim_converted, converted_parameters))(0, 0);

        SUBCASE("EvaluateModel function")
        {
            CHECK(m_low_dim.EvaluateModel(input_vec_low_dim) == doctest::Approx(output_low_dim));
            CHECK(m_squared_low_dim.EvaluateModel(input_vec_low_dim) == doctest::Approx(output_squared_low_dim));
            CHECK(m_high_dim.EvaluateModel(input_vec_high_dim) == doctest::Approx(output_high_dim));
        }

        SUBCASE("EvaluateLogModel function")
        {
            // Compute expected log outputs
            double output_low_dim_log = std::log(output_low_dim);
            double output_squared_low_dim_log = std::log(output_squared_low_dim);
            double output_high_dim_log = std::log(output_high_dim);

            CHECK(m_low_dim.EvaluateLogModel(input_vec_low_dim) == doctest::Approx(output_low_dim_log));
            CHECK(m_squared_low_dim.EvaluateLogModel(input_vec_low_dim) == doctest::Approx(output_squared_low_dim_log));
            CHECK(m_high_dim.EvaluateLogModel(input_vec_high_dim) == doctest::Approx(output_high_dim_log));
        }

        SUBCASE("EvaluateModelGrad function")
        {
            CHECK(m_low_dim.EvaluateModelGrad(input_vec_low_dim).isApprox(linear_fun_grad(input_vec_low_dim, parameters[0])));
            CHECK(m_squared_low_dim.EvaluateModelGrad(input_vec_low_dim).isApprox(squared_fun_grad(input_vec_low_dim, parameters[1])));
            CHECK(m_high_dim.EvaluateModelGrad(input_vec_high_dim).isApprox(component_sum_fun_grad(input_vec_high_dim, parameters[0])));
        }

        SUBCASE("EvaluateLogModelGrad function")
        {
            CHECK(m_low_dim.EvaluateLogModelGrad(input_vec_low_dim).isApprox(linear_fun_log_grad(input_vec_low_dim, parameters[0])));
            CHECK(m_squared_low_dim.EvaluateLogModelGrad(input_vec_low_dim).isApprox(squared_fun_log_grad(input_vec_low_dim, parameters[1])));
            CHECK(m_high_dim.EvaluateLogModelGrad(input_vec_high_dim).isApprox(component_sum_fun_log_grad(input_vec_high_dim, parameters[0])));
        }

        SUBCASE("EvaluateModelHessian function")
        {
            CHECK(m_low_dim.EvaluateModelHessian(input_vec_low_dim).isApprox(linear_fun_hess(input_vec_low_dim, parameters[0])));
            CHECK(m_squared_low_dim.EvaluateModelHessian(input_vec_low_dim).isApprox(squared_fun_hess(input_vec_low_dim, parameters[1])));
            CHECK(m_high_dim.EvaluateModelHessian(input_vec_high_dim).isApprox(component_sum_fun_hess(input_vec_high_dim, parameters[0])));
        }

        SUBCASE("EvaluateLogModelHessian function")
        {
            CHECK(m_low_dim.EvaluateLogModelHessian(input_vec_low_dim).isApprox(linear_fun_log_hess(input_vec_low_dim, parameters[0])));
            CHECK(m_squared_low_dim.EvaluateLogModelHessian(input_vec_low_dim).isApprox(squared_fun_log_hess(input_vec_low_dim, parameters[1])));
            CHECK(m_high_dim.EvaluateLogModelHessian(input_vec_high_dim).isApprox(component_sum_fun_log_hess(input_vec_high_dim, parameters[0])));
        }
    }

    SUBCASE("Parameter and model update")
    {
        Eigen::Vector2d p1(97.0, -8.9);
        Eigen::Matrix2d p2;
        p2 << 5.6, 0.199, -18.21, 1.22;
        parameters = {p1, p2};

        // Overwrite existing model to only use one parameter
        auto squared_fun_updated =
            [](const VectorXADd &x, const std::vector<MatrixXADd> &params) -> VectorXADd
        {
            return x.transpose() * params[0] * x;
        };

        m_low_dim.UpdateModel(squared_fun_updated);
        m_low_dim.UpdateParameters({p2});
        m_low_dim.Initialize();

        // Create another low dimensional model that only has one parameter
        auto linear_fun_updated =
            [](const VectorXADd &x, const std::vector<MatrixXADd> &params) -> VectorXADd
        {
            VectorXADd result(1);
            result << (x.array() * params[0].array()).sum();
            return result;
        };
        Model m_low_dim_another(low_dim);
        m_low_dim_another.UpdateModel(linear_fun_updated);
        m_low_dim_another.UpdateParameters({p1});
        m_low_dim_another.Initialize();

        // Check parameters
        CHECK(m_low_dim.GetParameters().size() == 1);
        CHECK(m_low_dim.GetParameters()[0].isApprox(p2));
        CHECK(m_low_dim_another.GetParameters().size() == 1);
        CHECK(m_low_dim_another.GetParameters()[0].isApprox(p1));

        // Define test input
        Eigen::Vector2d in(20.8, -9.2891);

        // Create sum model
        Model m_composed_sum(low_dim);
        m_composed_sum = m_low_dim + m_low_dim_another;

        // Compute expected outputs for the sum model
        double out_squared = ConvertFromCppAD(squared_fun_updated(ConvertToCppAD(in), {ConvertToCppAD(p2)}))(0, 0);
        double out_linear = ConvertFromCppAD(linear_fun_updated(ConvertToCppAD(in), {ConvertToCppAD(p1)}))(0, 0);

        Eigen::Vector2d out_squared_grad = squared_fun_grad(in, p2);
        Eigen::Vector2d out_linear_grad = linear_fun_grad(in, p1);

        Eigen::Matrix2d out_squared_hess = squared_fun_hess(in, p2);
        Eigen::Matrix2d out_linear_hess = linear_fun_hess(in, p1);

        Eigen::Matrix2d out_composed_sum_log_hess; // combined model result
        auto composed_sum_log_hess =
            [squared_fun_grad,
             linear_fun_grad,
             squared_fun_hess](const Eigen::VectorXd &x, const std::vector<Eigen::MatrixXd> &params) -> Eigen::MatrixXd
        {
            double denom = (x.transpose() * params[1] * x) + (params[0].array() * x.array()).sum();
            Eigen::Vector2d squared_grad_val = squared_fun_grad(x, params[1]); // hardcoded solution; hard to generalize to n-dimensions
            Eigen::Matrix2d squared_hess_val = squared_fun_hess(x, params[1]);
            Eigen::Vector2d linear_grad_val = linear_fun_grad(x, params[0]);

            Eigen::Matrix2d result; // hardcoded solution; hard to generalize to n-dimensions
            result << -std::pow(squared_grad_val(0) + linear_grad_val(0), 2),
                -(squared_grad_val(0) + linear_grad_val(0)) * (squared_grad_val(1) + linear_grad_val(1)),
                -(squared_grad_val(0) + linear_grad_val(0)) * (squared_grad_val(1) + linear_grad_val(1)),
                -std::pow(squared_grad_val(1) + linear_grad_val(1), 2);
            result /= std::pow(denom, 2);
            result += squared_hess_val / denom;
            return result;
        };
        out_composed_sum_log_hess = composed_sum_log_hess(in, parameters);

        // Check evaluation correctness
        CHECK(m_composed_sum.EvaluateModel(in) == doctest::Approx(out_squared + out_linear));
        CHECK(m_composed_sum.EvaluateLogModel(in) == doctest::Approx(std::log(out_squared + out_linear)));
        CHECK(m_composed_sum.EvaluateModelGrad(in).isApprox(out_squared_grad + out_linear_grad));
        CHECK(m_composed_sum.EvaluateLogModelGrad(in).isApprox((out_squared_grad + out_linear_grad) / (out_squared + out_linear)));
        CHECK(m_composed_sum.EvaluateModelHessian(in).isApprox(out_squared_hess + out_linear_hess));
        CHECK(m_composed_sum.EvaluateLogModelHessian(in).isApprox(out_composed_sum_log_hess));

        // Define new parameters to be updated
        Eigen::Vector2d in_new(-101.909, -0.5834);
        Eigen::Vector2d p1_new(-2.007, 83.03);
        Eigen::Matrix2d p2_new;
        p2_new << 6.92, -8.82, 27.1823, -0.6894;

        // Update new parameters
        m_composed_sum.UpdateParameters({p2_new, p1_new});
        m_composed_sum.Initialize();

        // Compute expected outputs for updated sum model
        out_squared = ConvertFromCppAD(squared_fun_updated(ConvertToCppAD(in_new), {ConvertToCppAD(p2_new)}))(0, 0);
        out_linear = ConvertFromCppAD(linear_fun_updated(ConvertToCppAD(in_new), {ConvertToCppAD(p1_new)}))(0, 0);

        out_squared_grad = squared_fun_grad(in_new, p2_new);
        out_linear_grad = linear_fun_grad(in_new, p1_new);

        out_squared_hess = squared_fun_hess(in_new, p2_new);
        out_linear_hess = linear_fun_hess(in_new, p1_new);

        out_composed_sum_log_hess = composed_sum_log_hess(in_new, {p1_new, p2_new});

        // Re-check evaluation correctness
        CHECK(m_composed_sum.EvaluateModel(in_new) == doctest::Approx(out_squared + out_linear));
        CHECK(m_composed_sum.EvaluateLogModel(in_new) == doctest::Approx(std::log(out_squared + out_linear)));
        CHECK(m_composed_sum.EvaluateModelGrad(in_new).isApprox(out_squared_grad + out_linear_grad));
        CHECK(m_composed_sum.EvaluateLogModelGrad(in_new).isApprox((out_squared_grad + out_linear_grad) / (out_squared + out_linear)));
        CHECK(m_composed_sum.EvaluateModelHessian(in_new).isApprox(out_squared_hess + out_linear_hess));
        CHECK(m_composed_sum.EvaluateLogModelHessian(in_new).isApprox(out_composed_sum_log_hess));
    }
}

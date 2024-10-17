#include "doctest.h"
#include "Core"
// #include <SVGDCpp/Core>
// #include <SVGDCpp/Model>

TEST_CASE("Model Class")
{
    int dimensions = 2;

    Model m_none;
    Model m_parametrized(dimensions);

    // Define a vector parameter
    Eigen::MatrixXd param_1(dimensions, 1);
    param_1 << 5.0, 7.0;

    // Define a matrix parameter
    Eigen::MatrixXd param_2(dimensions, dimensions);
    param_2 << 1.0, 3.0, 11.0, 13.0;

    // Define 2 parameters for the model
    std::vector<Eigen::MatrixXd> parameters(2);

    parameters[0] = param_1;
    parameters[1] = param_2;

    // Define sum of components function
    std::function<VectorXADd(const VectorXADd &x)> sum_of_comp_fun =
        [](const VectorXADd &x) -> VectorXADd
    {
        VectorXADd result(1);
        result << x.sum();
        return result;
    };

    // Define linear function
    std::function<VectorXADd(const VectorXADd &x)> linear_fun =
        [&m_parametrized](const VectorXADd &x) -> VectorXADd
    {
        VectorXADd result(1);
        result << (m_parametrized.GetParameters()[0].cast<CppAD::AD<double>>() * x).sum();
        return result;
    };

    // Define squared function
    std::function<VectorXADd(const VectorXADd &x)> squared_fun =
        [&m_parametrized](const VectorXADd &x) -> VectorXADd
    {
        VectorXADd result(1);
        return x.transpose() * m_parametrized.GetParameters()[1].cast<CppAD::AD<double>>() * x;
    };

    SUBCASE("Constructors")
    {
        SUBCASE("Default constructor")
        {
            CHECK(m_none.GetParameters().empty());
            CHECK(m_parametrized.GetParameters().empty());

            // Set model fuction
            m_parametrized.UpdateModel(sum_of_comp_fun);
            m_parametrized.UpdateParameters(parameters);

            for (size_t i = 0; i < parameters.size(); ++i)
            {
                CHECK(m_parametrized.GetParameters()[i].isApprox(parameters[i]));
            }
        }

        SUBCASE("Copy constructor")
        {
            // Model m_none_copied(m_none);
            // Model m_parametrized_copied(m_parametrized);
            // CHECK();

            // CHECK_THROWS_AS(m_none)
        }
    }

    SUBCASE("Operators")
    {
    }

    SUBCASE("Evaluate* functions")
    {
        SUBCASE("EvaluateModel function")
        {
        }

        SUBCASE("EvaluateModelGrad function")
        {
        }

        SUBCASE("EvaluateModelHessian function")
        {
        }

        SUBCASE("EvaluateLogModel function")
        {
        }

        SUBCASE("EvaluateLogModelGrad function")
        {
        }

        SUBCASE("EvaluateLogModelHessian function")
        {
        }
    }

    SUBCASE("Parameter update")
    {
    }

    SUBCASE("Model composition")
    {
    }

    SUBCASE("Edge cases")
    {
        // Exceptions
        SUBCASE("Exceptions")
        {
            // Without setting the model function, updating parameters should fail
            CHECK_THROWS_AS(m_parametrized.UpdateParameters(parameters), UnsetException);
        }
    }
}

#include <functional>

#include "doctest.h"
#include "Core"
#include "Kernel"
#include "Model"
#include "Optimizer"

/**
 * @brief Perform a manual SVGD step.
 *
 * @param num_particles Amount of particles.
 * @param x_in Input coordinate matrix.
 * @param kernel_fun Function object that computes the kernel value.
 * @param log_model_grad_fun Function object that computes the log model gradient.
 * @param kernel_grad_fun Function object that computes the kernel gradient.
 * @param optimizer_ptr Pointer to the optimizer object.
 * @param bounds Bounds for the problem.
 * @return Updated coordinate matrix.
 */
Eigen::MatrixXd ManualSVGDStep(const size_t &num_particles,
                               const Eigen::MatrixXd &x_in,
                               const std::vector<Eigen::MatrixXd> &model_parameters,
                               std::function<double(const Eigen::VectorXd &, const std::vector<Eigen::MatrixXd> &, const Eigen::VectorXd &)> kernel_fun,
                               std::function<Eigen::VectorXd(const Eigen::VectorXd &, const std::vector<Eigen::MatrixXd> &)> log_model_grad_fun,
                               std::function<Eigen::VectorXd(const Eigen::VectorXd &, const std::vector<Eigen::MatrixXd> &, const Eigen::VectorXd &)> kernel_grad_fun,
                               const std::shared_ptr<Optimizer> &optimizer_ptr,
                               const std::pair<Eigen::VectorXd, Eigen::VectorXd> &bounds)
{
    Eigen::MatrixXd phi = Eigen::MatrixXd::Zero(x_in.rows(), x_in.cols());
    Eigen::MatrixXd x_out = Eigen::MatrixXd::Zero(x_in.rows(), x_in.cols());

    for (size_t i = 0; i < num_particles; ++i)
    {
        Eigen::VectorXd curr_particle_coords = x_in.col(i); // i-th particle coordinate

        for (size_t j = 0; j < num_particles; ++j)
        {
            // Compute kernel value
            double kernel_val = kernel_fun(x_in.col(j), {}, curr_particle_coords);

            // Compute gradient values
            Eigen::VectorXd log_model_grad_vec = log_model_grad_fun(x_in.col(j), model_parameters);
            Eigen::VectorXd kernel_grad_vec = kernel_grad_fun(x_in.col(j), {}, curr_particle_coords);

            // Sum the gradient value
            phi.col(i) += kernel_val * log_model_grad_vec + kernel_grad_vec;
        }
    }

    phi /= num_particles;

    x_out = x_in + optimizer_ptr->Step(phi);

    // Check bounds manually
    for (int i = 0; i < x_out.rows(); ++i)
    {
        x_out.row(i) = (x_out.row(i).array() < bounds.first(i)).select(bounds.first(i), x_out.row(i));
        x_out.row(i) = (x_out.row(i).array() > bounds.second(i)).select(bounds.second(i), x_out.row(i));
    }

    return x_out;
}

TEST_CASE("SVGD class")
{
    // Compute SVGD manually and compare

    // Define parameters
    size_t dimension = 2, num_particles = 10, num_iterations = 15;

    Eigen::Vector2d lower_bound(-1.0, -1.0);
    Eigen::Vector2d upper_bound(1.0, 1.0);

    // Define Model:
    // a * cos(x[0]) + b * cos(x[1]) + c * x[0] * x[1] - d
    // a = 7.5, b = 10, c = 3, d = -6
    std::function<VectorXADd(const VectorXADd &, const std::vector<MatrixXADd> &)> model_fun_ad =
        [](const VectorXADd &x, const std::vector<MatrixXADd> &params)
    {
        VectorXADd result(1);
        result << params[0] * CppAD::cos(x(0)) + params[1] * CppAD::cos(x(1)) + params[2] * x.prod() + params[3];
        return result;
    };

    std::vector<Eigen::MatrixXd> model_parameters{Eigen::VectorXd::Constant(1, 7.5),
                                                  Eigen::VectorXd::Constant(1, 10.0),
                                                  Eigen::VectorXd::Constant(1, 3.0),
                                                  Eigen::VectorXd::Constant(1, -6.0)};

    std::shared_ptr<Model> model_ptr = std::make_shared<Model>(dimension);
    model_ptr->UpdateModel(model_fun_ad);
    model_ptr->UpdateParameters(model_parameters);

    // Define Kernel:
    // exp( -||x - x'||^2 )
    std::function<VectorXADd(const VectorXADd &, const std::vector<MatrixXADd> &, const VectorXADd &)> kernel_fun_ad =
        [](const VectorXADd &x, const std::vector<MatrixXADd> &params, const VectorXADd &location)
    {
        VectorXADd result(1), diff = x - location;
        result << (-diff.transpose() * diff).array().exp();
        return result;
    };

    std::shared_ptr<Kernel> kernel_ptr = std::make_shared<Kernel>(dimension);
    kernel_ptr->UpdateKernel(kernel_fun_ad);

    // Define Adam optimizer
    std::shared_ptr<Optimizer>
        optimizer_ptr = std::make_shared<Adam>(dimension, num_particles, 1.0e-1, 0.9, 0.999);

    // Define arbitrary starting coordinates for particles
    std::shared_ptr<Eigen::MatrixXd> x_matrix = std::make_shared<Eigen::MatrixXd>(Eigen::MatrixXd::Random(dimension, num_particles));

    // Create a copy of the matrix
    Eigen::MatrixXd x_matrix_copy_initial = *x_matrix;

    // Instantiate the SVGD class
    SVGDOptions options;
    options.Dimension = dimension;
    options.NumIterations = num_iterations;
    options.CoordinateMatrixPtr = x_matrix;
    options.KernelPtr = kernel_ptr;
    options.ModelPtr = model_ptr;
    options.OptimizerPtr = optimizer_ptr;
    options.LowerBound = lower_bound;
    options.UpperBound = upper_bound;
    options.LogIntermediateMatrices = true;

    SVGD svgd(options);

    // Execute and collect data
    svgd.Initialize();
    svgd.Run();

    CHECK(x_matrix->size() == x_matrix_copy_initial.size());
    CHECK(x_matrix->rows() == x_matrix_copy_initial.rows());
    CHECK(x_matrix->cols() == x_matrix_copy_initial.cols());

    /*
        Manual calculations
    */

    // Create optimizer copy
    std::shared_ptr<Optimizer> optimizer_copy_ptr = std::make_shared<Adam>(dimension, num_particles, 1.0e-1, 0.9, 0.999);
    optimizer_copy_ptr->Initialize();

    // Define double-typed kernel function
    auto kernel_fun = [kernel_fun_ad](const Eigen::VectorXd &x,
                                      const std::vector<Eigen::MatrixXd> &params,
                                      const Eigen::VectorXd &location) -> double
    {
        std::vector<MatrixXADd> params_ad;
        std::transform(params.begin(), params.end(), params_ad.begin(), ConvertToCppAD);
        return ConvertFromCppAD(kernel_fun_ad(ConvertToCppAD(x),
                                              params_ad,
                                              ConvertToCppAD(location)))(0);
    };

    // Define double-typed kernel gradient function
    auto kernel_grad_fun = [](const Eigen::VectorXd &x,
                              const std::vector<Eigen::MatrixXd> &params,
                              const Eigen::VectorXd &location) -> Eigen::VectorXd
    {
        Eigen::VectorXd diff = x - location;
        Eigen::VectorXd result = diff;
        return result * -2 * std::exp(-diff.transpose() * diff);
    };

    // Define double-typed log model gradient function
    auto log_model_grad_fun = [model_fun_ad](const Eigen::VectorXd &x,
                                             const std::vector<Eigen::MatrixXd> &params) -> Eigen::VectorXd
    {
        std::vector<MatrixXADd> params_ad(params.size());
        std::transform(params.begin(), params.end(), params_ad.begin(), ConvertToCppAD);
        double denominator = ConvertFromCppAD(model_fun_ad(ConvertToCppAD(x),
                                                           params_ad))(0);

        Eigen::VectorXd result(2);
        result << -params[0] * std::sin(x(0)) + params[2] * x(1), -params[1] * std::sin(x(1)) + params[2] * x(0);
        return result / denominator;
    };

    // Perform manual computation
    Eigen::MatrixXd x_matrix_copy_final = x_matrix_copy_initial;

    // Check that the initial results are not the same
    CHECK(!x_matrix->isApprox(x_matrix_copy_final));

    for (size_t i = 0; i < num_iterations; ++i)
    {
        x_matrix_copy_final = ManualSVGDStep(num_particles,
                                             x_matrix_copy_final,
                                             model_parameters,
                                             kernel_fun,
                                             log_model_grad_fun,
                                             kernel_grad_fun,
                                             optimizer_copy_ptr,
                                             std::pair(lower_bound, upper_bound));
    }

    // Check that the final results are the same
    CHECK(x_matrix->isApprox(x_matrix_copy_final));
}
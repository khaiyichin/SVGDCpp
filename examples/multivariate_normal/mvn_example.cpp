#include <iostream>
#include "Core"
#include "Model"
#include "Kernel"
#include "Optimizer"

int main()
{
    Eigen::Vector2d mean(-0.6871, 0.8010);

    Eigen::Matrix2d covariance;

    covariance << 0.2260, 0.1652, 0.1652, 0.6779;
    covariance *= 5;

    std::shared_ptr<Model> mvn_ptr = std::make_shared<MultivariateNormal>(mean, covariance);

    // Create particles
    size_t dim = 2;
    size_t num_particles = 10;
    size_t num_iterations = 1000;

    auto x0 = std::make_shared<Eigen::MatrixXd>(3 * Eigen::MatrixXd::Random(dim, num_particles));

    std::cout << "Initial particle coordinates" << std::endl
              << *x0 << std::endl;

    // Instantiate a kernel function
    std::shared_ptr<Kernel> rbf_ptr = std::make_shared<GaussianRBFKernel>(x0, GaussianRBFKernel::ScaleMethod::Median, mvn_ptr);

    // Instantiate an optimizer
    std::shared_ptr<Optimizer> opt_ptr = std::make_shared<AdaGrad>(dim, num_particles, 1.0e-1);

    // Instantiate the SVGD class
    SVGD svgd(dim, num_iterations, x0, rbf_ptr, mvn_ptr, opt_ptr);

    // Initialize and run SVGD
    svgd.Initialize();
    svgd.Run();

    std::cout << "Final particle coordinates" << std::endl
              << *x0 << std::endl;
}

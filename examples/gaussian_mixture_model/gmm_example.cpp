#include <iostream>
#include <SVGDCpp/Core>
#include <SVGDCpp/Model>
#include <SVGDCpp/Kernel>
#include <SVGDCpp/Optimizer>

int main()
{
    Eigen::Vector2d mean1(3.6871, -2.801), mean2(-2.9802, 4.3387);
    
    Eigen::Matrix2d cov1, cov2;
    
    cov1 << 0.5001, 0.2426, 0.2426, 0.8420;
    cov2 << 0.6779, -0.1652, -0.1652, 0.2260;

    cov1 *= 5;
    cov2 *= 5;

    // Define 2 MVNs
    MultivariateNormal mvn1(mean1, cov1);
    MultivariateNormal mvn2(mean2, cov2);

    // Create GMM
    Model gmm = mvn1 + mvn2;

    std::shared_ptr<Model> gmm_ptr = std::make_shared<Model>(gmm);

    // Create particles
    size_t dim = 2;
    size_t num_particles = 20;
    size_t num_iterations = 1000;

    auto x0 = std::make_shared<Eigen::MatrixXd>(8 * Eigen::MatrixXd::Random(dim, num_particles));
    
    std::cout << "Initial particle coordinates" << std::endl
              << *x0 << std::endl;

    // Instantiate a kernel function
    std::shared_ptr<Kernel> rbf_ptr = std::make_shared<GaussianRBFKernel>(x0, GaussianRBFKernel::ScaleMethod::Median, gmm_ptr);

    // Instantiate an optimizer
    std::shared_ptr<Optimizer> opt_ptr = std::make_shared<Adam>(dim, num_particles, 1.0e-1, 0.9, 0.999);
    
    // Instantiate the SVGD class
    SVGD svgd(dim, num_iterations, x0, rbf_ptr, gmm_ptr, opt_ptr);

    // Initialize and run SVGD
    svgd.Initialize();
    svgd.Run();

    std::cout << "Final particle coordinates" << std::endl
              << *x0 << std::endl;
}

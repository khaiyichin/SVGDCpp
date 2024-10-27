# SVGDCpp: An SVGD library in C++

WORK-IN-PROGRESS

- install eigen: from source or apt install libeigen3-dev
- doxygen required for BUILD_DOCUMENTATION=TRUE (FALSE by default)
- to build tests you need BUILD_TESTS=TRUE (FALSE by default)
- to build examples you need BUILD_EXAMPLES=TRUE (FALSE by default)
- configure cppad with the `-D cppad_testvector=eigen`, `include_eigen=true`, and `-D CMAKE_BUILD_TYPE=Release` flags

- openmp version 201511 == v4.5

TODO:
- figure out how to incorporate model_parameters_ into AD function == `UpdateParameters` and `ModelFun` will do that now
- figure out whether a parameter can be changed and differentiation will still happen without `SetupADFun()` == no it doesn't update, so differentiation happens wrt to old variable

## How to use as part of CMake projects?
```cmake
find_package(SVGDCpp REQUIRED)

add_executable(test_svgd test_svgd.cpp)

target_link_libraries(test_svgd
    PRIVATE
    SVGDCpp::SVGDCpp
)
```

Then in source code:
```cpp
#include <SVGDCpp/Core>
#include <SVGDCpp/Model>
#include <SVGDCpp/Kernel>
#include <SVGDCpp/Optimizer>
```

## Example usage
```cpp
// Set up a 2-D multivariate normal problem with a RBF kernel using 10 particles and the Adam optimizer for 1000 iterations
size_t dim = 2, num_particles = 10, num_iterations = 1000;
auto x0 = std::make_shared<Eigen::MatrixXd>(3*Eigen::MatrixXd::Random(dim, num_particles));

// Create multivariate normal model pointer
Eigen::Vector2d mean(5, -5);
Eigen::Matrix2d covariance;
covariance << 0.5, 0, 0, 0.5;
std::shared_ptr<Model> model_ptr = std::make_shared<MultivariateNormal>(mean, covariance);

// Create RBF kernel pointer
std::shared_ptr<Kernel> kernel_ptr = std::make_shared<GaussianRBFKernel>(x0, GaussianRBFKernel::ScaleMethod::Median, model_ptr);

// Create Adam optimizer pointer
std::shared_ptr<Optimizer> opt_ptr = std::make_shared<Adam>(dim, num_particles, 1.0e-1, 0.9, 0.999);

// Instantiate the SVGD class
SVGD svgd(dim, num_iterations, x0, kernel_ptr, model_ptr, opt_ptr);

// Initialize and run SVGD for 1000 iterations
svgd.Initialize();
svgd.Run();
```
<!-- Note that the `Initialize()` method should be called only through the `SVGD` class instance. The `Model` and `Kernel` classes provide an `Initialize()` method but are only used if you want to use them for computation manually. -->

### Using the Kernel and Model class for manual computation (advanced)
Whenever a model or a kernel is defined for the first time, they're uninitialized, so you'll need to initialize them if you want to use them directly. This applies if you're copying from an existing object.
```cpp
// Create RBF kernel pointer
std::shared_ptr<Kernel> kernel_ptr = std::make_shared<GaussianRBFKernel>(x0, GaussianRBFKernel::ScaleMethod::Median, model_ptr);

// Compute kernel value with arbitrary values and at the default location (i.e., reference coordinate)
kernel_ptr->Initialize(); // initialize the kernel so that the `Evaluate*` functions are available
Eigen::Vector2d input(16.328, -4.059);
double output = kernel_ptr->EvaluateKernel(input);

// Compute kernel value with arbitrary values and at an arbitrary location
Eigen::Vector2d new_location(5.2225, 97.6);
kernel_ptr->UpdateLocation(new_location);
kernel_ptr->Initialize();
output = kernel_ptr->EvaluateKernel(input);
```

WARNING: anytime an `Update*` method is called _manually_, the instance's `Initialize()` function must be called before the scope ends. This is so that the `CppAD` tape can record any modifications to the functions. E.g.,
```cpp
// Okay
{
    Model m;
    m.UpdateModel(some_fun);
    m.Initialize();
}

// Also okay (order of Update* doesn't matter here)
{
    Model m;
    m.UpdateParameters(some_params);
    m.UpdateModel(some_fun);
    m.Initialize();
}

// Okay, but redundant
{
    Model m;
    m.UpdateParameters(some_params);
    m.Initialize();
    m.UpdateModel(some_fun);
    m.Initialize();
}

// Applies also to Kernel classes (order doesn't matter here, as long as `Initialize()` is called at the end)
{
    Kernel k;
    k.UpdateKernel(some_fun);
    k.UpdateLocation(some_vec);
    k.UpdateParameters(some_params);
    k.Initialize();

    k.UpdateParameters(some_new_params); // doesn't get registered; subsequent computation uses the `some_params` values
}

// If you defined models/kernels to be composed into a new model (and don't intend to use the original objects), then `Initialize()` is only needed for the new object
{
    Model m_a(2), m_b(2);
    m_a.UpdateModel(model_function_a);
    m_b.UpdateModel(model_function_b);
    Model m_new(2);
    m_new = m_a + m_b;
    m_new.Initialize();

    // same for copied objects
    Model m_another(m_new);
    m_another.Initialize(); // required
}
```
<!-- The only exception is if you create a derived class (see below): do not call `Initialize()` within any of the derived class constructors, regardless if you call any `Update*` functions. You can still call them in other places within the class, so as long that **they're not invoked during the construction phase**. -->

<!-- The reason behind this is due to the way CppAD (the autodiff library) records the operation sequence in parallel mode: the same thread that initially handles the recording must remain the same manager. So if `Initialize()` is called outside of parallel mode (_i.e.,_ by thread #0) then subsequent calls of `Initialize()` by other threads will cause execution to fail. -->

Typically, unless you call the model and kernel `Update*` methods directly, you don't need to worry about calling `Initialize()` yourself. In most cases, you interact with the `SVGD` object, and its `UpdateModelParameters` and `UpdateKernelParameters` functions take care of re-initializing the updates internally.

WARNING: (maybe not a problem since SVGD creates copies of the instance?) using SVGD

### How to create a custom kernel or model?

Note: the model **MUST** be a non-negative scalar function (if negative model function then log will cause program to crash). The model function is essentially the posterior probability density function to be estimated.

Note 2: By definition, a kernel function must be specified such that the matrix generated by the said function must be positive definite (ref: [arXiV: Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm](https://arxiv.org/abs/1608.04471) and [Wikipedia: RKHS](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space))

Method 1: using the Kernel or Model class directly. This is useful if you want to compose different models into one.
```cpp

// Create 2 models, f1 = a1 * x and f2 = a2 + exp(b * x)
std::function<VectorXADd(const VectorXADd &x, const std::vector<MatrixXADd> &params)> model_fun1 = [](const VectorXADd &x, const std::vector<MatrixXADd> &params)
    {
        Eigen::VectorXd mean(2);
        mean << 1,1;
        Eigen::MatrixXd cov(2,2);
        cov << 0.5, 0, 0, 0.5;
        VectorXADd diff = x - mean.cast<CppAD::AD<double>>();
        return VectorXADd((- (diff.transpose() * cov.inverse().cast<CppAD::AD<double>>() * diff)).array().exp());
    };
std::function<VectorXADd(const VectorXADd &x)> model_fun = [](const VectorXADd &x)
    {
        VectorXADd temp(1);
        temp << x.sum();
        return temp;
    };
Model model1 = Model(2);
model1.UpdateModel(model_fun);
Model model2 = Model(2);
model2.UpdateModel([](const VectorXADd &x) {VectorXADd temp(1); temp << 2*x.sum(); return temp;});
Model combined = model1 + model2;

Eigen::VectorXd temp(2);
temp << 0.5, 2.6;

// Initialize models (REQUIRED IF YOU WANT TO EVALUATE THE FUNCTIONS DIRECTLY; OTHERWISE SVGD::Initialize does that automatically for you)
model1.Initialize();
model2.Initialize();
combined.Initialize();
```
NOTE: `UpdateModel` must be done before `UpdateParameters` can be executed in this case


(if you need automatic differentiation and want to write your own derived class)
Method 2: create a derived class from Kernel or Model (if you need automatic differentiation and want to write your own derived class). Ensure also the `Clone*` functions are provided for polymorphic copy which is required for multi-threaded execution of SVGD (you may neglect that if you only wish to run SVGD in sequential mode).

(NOT RECOMMENDED TO MANUALLY OVERRIDE `ModelFun`; create a `std::function` and feed into `UpdateModel` instead. This is because overriding `ModelFun` directly means the class silently removes the ability to update the model using `UpdateModel`)
```cpp

class MultivariateNormal : public Model
{
public:
    MultivariateNormal() {}

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
        auto model_fun = [](const VectorXADd &x, const std::vector<MatrixXADd> &params)
        {
            VectorXADd result(1), diff = x - params[0];
            result << (-0.5 * (diff.transpose() * params[1].inverse() * diff).array()).exp();
            return result;
        };

        UpdateModel(model_fun);
    }

    // Provide polymorphic copying functionality (REQUIRED)
    virtual std::unique_ptr<Model> CloneUniquePointer() const
    {
        return std::make_unique<MultivariateNormal>(*this);
    }

    // Provide polymorphic copying functionality (REQUIRED)
    virtual std::shared_ptr<Model> CloneSharedPointer() const
    {
        return std::make_shared<MultivariateNormal>(*this);
    }

    /* and so on */
};
```
Method 2b (advanced usage): override `EvaluateModel*` and/or `EvaluateLogModel*` directly if you don't need automatic differentiation AND if you don't intend to compose new models from these derived models. This is because functional composition (e.g., obj1+obj2, obj1*obj2) utilize the `ModelFun` method of the respective `Model` objects; overriding the `Evaluate*` methods directly means combining the `nullptr`s of both `model_fun_` variables. THIS IS USEFUL IF YOU CAN PROVIDE CLOSED-FORM FUNCTIONS BECAUSE IT PROVIDES SOME OPTIMIZATION (THOUGH I'M NOT SURE BY HOW MUCH).

### Location definition
the location in the Kernel class refers to the second argument in the kernel function k(x, x'). The first argument will be the one that the gradient is taken with respect to.

## Some common exceptions thrown by CppAD
```
cppad-20240602 error from a known source:
yq = f.Forward(q, xq): a zero order Taylor coefficient is nan.
Corresponding independent variables vector was written to binary a file.
vector_size = 2
```
source: invalid evaluation of function, e.g., log of a negative number, division by zero, etc.

## Alternatives of common math functions for CppAD::AD<double> types
| Instead of | Use |
| -- | -- |
| `std::exp` | `CppAD::exp` |
| `std::abs` | `CppAD::abs` |
| `std::pow` | `CppAD::pow` |
| `std::sqrt` | `CppAD::sqrt` |

## Multi-threaded execution
Running SVGD in parallel mode is as simple as calling `SetupForParallelMode()` and then instantiating the `SVGD` class with the `parallel` argument as `true`.
```cpp
// Set up parallel mode execution
SetupForParallelMode();

/* define dim, num_iterations, x0m, model_ptr, kernel_ptr, opt_ptr */

// Instantiate the SVGD class with the `parallel` argument as true
SVGD svgd(dim, num_iterations, x0, kernel_ptr, model_ptr, opt_ptr, true);
svgd.Initialize();
```
If activated, it will run with the maximum number of threads the machine has. Note: you're unlikely to see performance improvements if you have a small number of particles (n <= 10).
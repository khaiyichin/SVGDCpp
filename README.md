# SVGDCpp: An SVGD library in C++

WORK-IN-PROGRESS

install eigen: from source or apt install libeigen3-dev
doxygen required for BUILD_DOCUMENTATION=TRUE (FALSE by default)
to build tests you need BUILD_TESTS=TRUE (FALSE by default)
to build examples you need BUILD_EXAMPLES=TRUE (FALSE by default)
configure cppad with the `-D cppad_testvector=eigen`, `include_eigen=true`, and `-D CMAKE_BUILD_TYPE=Release` flags


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

WARNING: anytime an `Update*` method is called _manually_, the instance's `Initialize()` function must be called before the scope ends. E.g.,
```
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

// Applies also to Kernel classes
{
    Kernel k;
    k.UpdateKernel(some_fun);
    k.Initialize();
}
```
Typically, unless you call the model and kernel `Update*` methods directly, you don't need to worry about calling `Initialize()` yourself. In most cases, you interact with the `SVGD` object, and its `UpdateModelParameters` and `UpdateKernelParameters` functions take care of re-initializing the updates internally.

## How to create a kernel or model?

Note: the kernel and model **MUST** to be scalar functions

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
Method 2: create a derived class from Kernel or Model (if you need automatic differentiation and want to write your own derived class). (NOT RECOMMENDED TO MANUALLY OVERRIDE `ModelFun`; create a `std::function` and feed into `UpdateModel` instead. This is because overriding `ModelFun` directly means the class silently removes the ability to update the model using `UpdateModel`)
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
            throw std::runtime_error("Dimensions of parameter vectors/matrices do not match.");
        }

        // Store parameters
        model_parameters_.push_back(mean);
        model_parameters_.push_back(covariance);

        // Compute the normalization constant based on the updated parameters
        ComputeNormalizationConstant();

        // Define model function (the kernel density only, without normalization constant)
        auto model_fun = [this](const VectorXADd &x)
        {
            VectorXADd result(1), diff = x - this->model_parameters_[0].cast<CppAD::AD<double>>();
            result << (-0.5 * (diff.transpose() * this->model_parameters_[1].cast<CppAD::AD<double>>().inverse() * diff).array()).exp();
            return result;
        };

        UpdateModel(model_fun);
    }

    /* and so on */
};
```
Method 2b (advanced usage): override `EvaluateModel*` and/or `EvaluateLogModel*` directly if you don't need automatic differentiation AND if you don't intend to compose new models from these derived models. This is because functional composition (e.g., obj1+obj2, obj1*obj2) utilize the `ModelFun` method of the respective `Model` objects; overriding the `Evaluate*` methods directly means combining the `nullptr`s of both `model_fun_` variables. THIS IS USEFUL IF YOU CAN PROVIDE CLOSED-FORM FUNCTIONS BECAUSE IT PROVIDES SOME OPTIMIZATION (THOUGH I'M NOT SURE BY HOW MUCH).


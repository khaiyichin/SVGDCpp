install eigen: from source or apt install libeigen3-dev
configure cppad with the `-D cppad_testvector=eigen`, `include_eigen=true`, and `-D CMAKE_BUILD_TYPE=Release` flags


TODO:
- figure out how to incorporate model_parameters_ into AD function
- figure out whether a parameter can be changed and differentiation will still happen without `SetupADFun()` == no it doesn't update, so differentiation happens wrt to old variable


How to create a kernel or model?

Note: the kernel and model have to be scalar functions

Method 1: using the Kernel or Model class directly.
```cpp

// Create 2 models, f1 = a1 * x and f2 = a2 + exp(b * x)
std::function<VectorXADd(const VectorXADd &x)> model_fun1 = [](const VectorXADd &x)
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
        temp << x.array().sum();
        return temp;
    };
Model model1 = Model(2);
model1.UpdateModel(model_fun);
Model model2 = Model(2);
model2.UpdateModel([](const VectorXADd &x) {VectorXADd temp(1); temp << 2*x.array().sum(); return temp;});
Model combined = model1 + model2;

Eigen::VectorXd temp(2);
temp << 0.5, 2.6;


model1.Initialize();
model2.Initialize();
combined.Initialize();
```

Method 2: create a derived class from Kernel or Model.
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
    }

    ~MultivariateNormal() {}

    MultivariateNormal &operator=(const MultivariateNormal &obj)
    {
        mean_vec_ad_ = obj.mean_vec_ad_;
        cov_mat_ad_ = obj.cov_mat_ad_;

        Model::operator=(obj);

        return *this;
    }

    void UpdateParameters(const std::vector<Eigen::MatrixXd> &params) override
    {
        Eigen::VectorXd mean = params[0];
        Eigen::MatrixXd covariance = params[1];

        // Ensure that the dimensions of mean matches covariance
        if (!CompareVectorSizes<Eigen::VectorXd, Eigen::VectorXd>(mean, covariance.col(0)) ||
            !CompareVectorSizes<Eigen::VectorXd, Eigen::VectorXd>(mean, covariance.row(0)))
        {
            throw std::runtime_error("Dimensions of parameter vectors/matrices do not match.");
        }

        mean_vec_ad_ = mean.cast<CppAD::AD<double>>();
        cov_mat_ad_ = covariance.cast<CppAD::AD<double>>();

        // Compute the normalization constant based on the updated parameters
        ComputeNormalizationConstant();

        Initialize();
    }

protected:
    VectorXADd ModelFun(const VectorXADd &x) const override
    {
        VectorXADd diff = x - model_parameters_[0].cast<CppAD::AD<double>>();
        return (-0.5 * (diff.transpose() * model_parameters_[1].cast<CppAD::AD<double>>().inverse() * diff).array()).exp();
    }
};
```

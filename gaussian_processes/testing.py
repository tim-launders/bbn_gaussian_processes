# Testing file for custom GP implementation

import numpy as np
import jax.numpy as jnp
from jax import grad
import matplotlib.pyplot as plt
from jax.random import PRNGKey, normal, multivariate_normal, split
from kernels import Constant, RBF, Dot
import gp
import optimizers

num_datasets = 1
num_datapoints = 4
percent_error = 0.1
percent_correlation = 0

# Generate synthetic data

all_X = []
all_y = []
all_total_error = []
all_normalization_error = []
points = []

for i in range(num_datasets):
    j=i
    key = PRNGKey(j)
    all_X.append(1 + jnp.linspace(i, 10+i, (num_datapoints+i)))
    all_y.append(1 + jnp.linspace(i, 10+i, (num_datapoints+i)) + normal(key=key, shape=(num_datapoints+i,)))
    all_total_error.append(percent_error * (i+1) * jnp.abs(all_y[i]))
    all_normalization_error.append(percent_correlation * all_total_error[i])
    points.append(num_datapoints+i)
X = jnp.concatenate(all_X)
y = jnp.concatenate(all_y)
total_error = jnp.concatenate(all_total_error)
normalization_error = jnp.concatenate(all_normalization_error)


#kernel = Constant(1.0) * RBF(1.0, prior_type='log_normal') + Constant(10.0, prior_type='fixed') * Dot(1.0)
kernel = Dot(1e-2, prior_type='uniform')
gpcov = gp.GPR(kernel)
cov_mat = gpcov.calculate_covariance(total_error, normalization_error, points)

print(kernel.get_params())

optimizer = optimizers.MarginalLikelihoodOptimizer(kernel, X, y, cov_mat, points=points)
print(kernel.get_params())
print(optimizer.loss(kernel.get_params()))

params, min_loss = optimizer.adam(1e-4, 1e-2, 0.9, 0.999)
print(params)
print(min_loss)

kernel.set_params(params)

gpr = gp.GPR(kernel=kernel)
gpr.fit(X, y, total_error=total_error, normalization_error=normalization_error, points=points)
print("Hyperparameters after fitting:", gpr.get_kernel_hyperparameters())

X_test = jnp.linspace(-5, max(X)+0.5, 100)

y_pred, sigma_pred = gpr.predict(X_test, return_std=True)

samples = gpr.draw_samples(X_test, 5, seed=1)

for i in range(num_datasets):
    plt.errorbar(all_X[i], all_y[i], all_total_error[i], fmt='.')

plt.plot(X_test, y_pred, label='Prediction')
plt.fill_between(X_test, y_pred - sigma_pred, y_pred + sigma_pred, alpha=0.5, label=r'1$\sigma$ confidence interval')

for sample in samples:
    plt.plot(X_test, sample)

plt.legend()
plt.show()

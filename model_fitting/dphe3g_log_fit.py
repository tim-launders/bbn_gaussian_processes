# Calculates the best-fit hyperparameters for a kernel to fit the ddHe3n S-factor data


import sys
sys.path.append('.')

import jax
import numpy as np
import jax.numpy as jnp
from scipy.interpolate import UnivariateSpline
from gaussian_processes.kernels import Constant, SE, Dot, Matern, RationalQuadratic
from gaussian_processes.optimizers import MarginalLikelihoodOptimizer, LOOOptimizer, kfoldCVOptimizer, LeaveDatasetOutOptimizer
from gaussian_processes.gp import GPR
import matplotlib.pyplot as plt

# Import S-factor data

jax.config.update('jax_enable_x64', True)

data_dir = './data/dpHe3g/'

sets = ['GR62', 'GR63', 'WA63', 'MA97', 'SC97', 'CA02', 'TI19', 'MO20', 'TU21 3', 'TU21 4']
colors = ['indigo', 'seagreen', 'brown', 'cyan', 'red', 'orange', 'grey', 'blue', 'darkslateblue', 'magenta']

all_E = []
all_S = []
all_total_error = []
all_systematic_error = []
all_statistical_error = []
points = []

for set in sets:
    energy, cross, Sfactor, total, sys = np.loadtxt(data_dir+set+'.txt', unpack=True)
    all_E.append(energy)
    all_S.append(Sfactor)
    all_total_error.append(total)
    all_systematic_error.append(sys)
    all_statistical_error.append(np.sqrt(total**2 - sys**2))
    points.append(len(Sfactor))

E = np.concatenate(all_E)
logE = np.log(E)
S = np.concatenate(all_S)
logS = np.log(S)
total_error = np.concatenate(all_total_error)
systematic_error = np.concatenate(all_systematic_error)
statistical_error = np.concatenate(all_statistical_error)

def lognormal_var(error, mean):
    return np.sqrt(np.log((error/mean)**2+1))

log_total_error = lognormal_var(total_error, S)
log_systematic_error = lognormal_var(systematic_error, S)
log_statistical_error = lognormal_var(statistical_error, S)


# Load in theory curve

file = './data/dphe3g_theory_curve.csv'
theory_E, theory_S = np.loadtxt(file, delimiter=',', unpack=True)
sort_idx = np.argsort(theory_E)
x_sorted = np.log(theory_E[sort_idx])
y_sorted = theory_S[sort_idx]
theory_mean_base = UnivariateSpline(x=x_sorted, y=y_sorted, s=0)
theory_mean = lambda X: jnp.log(jnp.array([theory_mean_base(x) if x > x_sorted[0] else y_sorted[0] for x in X]))
theory_mean = lambda X : jnp.zeros_like(X)


hyper = [2.31789296E+03, 7.60324923E+00, 2.66962082E-02, 3.74584365E+02]

global_kernel = Constant(hyper[0], prior_params=[1e-13,1e10]) * SE(hyper[1], prior_params=[1e-6,1e8]) 
local_kernel = Constant(hyper[2], prior_params=[1e-16,1e6]) * Matern(hyper[3], nu=0.25, prior_params=[1e-6,1e14])
kernel = global_kernel + local_kernel

gpcov = GPR(kernel)
cov_mat = gpcov.calculate_covariance(log_total_error, log_systematic_error, points)

# Add dependencies between TU21 datasets
scale_mat = np.zeros_like(cov_mat)
start_ind = np.sum(points[0:-2])
scale_factor = 0.12
for i in range(points[-2]):
    for j in range(points[-1]):
        row = start_ind + i
        col = start_ind + points[-2] + j
        S_row = S[row]
        S_col = S[col]
        row_error = scale_factor * S_row
        col_error = scale_factor * S_col
        log_row_error = lognormal_var(row_error, S_row)
        log_col_error = lognormal_var(col_error, S_col)
        scale_mat[row, col] += log_row_error * log_col_error
        # Repeat for symmetric part
        scale_mat[col, row] = scale_mat[row, col]
cov_mat += scale_mat

S_subtract = logS - theory_mean(logE)

E_pred = jnp.logspace(np.log10(2.5e-3), np.log10(2), 300)

optimizer = LeaveDatasetOutOptimizer(kernel, logE, S_subtract, cov_mat, points=points)

print('Initial loss: ', optimizer.loss(jnp.log(kernel.get_params())))
"""
params, min_loss = optimizer.adam(tolerance=1e-5)
print('\nFinal loss: ', min_loss)
print('Optimized hyperparameter values: ', params)
kernel.set_params(params)
"""
gp = GPR(kernel, noise=0, mean_func=theory_mean)
gp.fit(logE, logS, log_total_error, log_systematic_error, points)
gp.set_covariance(cov_mat)

num_points = 200

E_pred = jnp.logspace(np.log10(2.5e-3), np.log10(2), num_points)

log_mean, log_sigma = gp.predict(jnp.log(E_pred), True)
mean = jnp.exp(log_mean)
sigma_up = jnp.exp(log_mean + log_sigma)
sigma_down = jnp.exp(log_mean - log_sigma)

for i in range(len(points)):
    plt.errorbar(all_E[i], 1e6*all_S[i], 1e6*all_total_error[i], fmt=".", color=colors[i], label=sets[i])
plt.plot(E_pred, 1e6*mean, color='black', label='GPR Mean')
plt.fill_between(E_pred, 1e6*sigma_down, 1e6*sigma_up, alpha=0.4, linewidth=0, color='black')

samples = gp.draw_samples(jnp.log(E_pred), 3, seed=5)
for sample in samples: 
    sample = 1e6*jnp.exp(sample)
    plt.plot(E_pred, sample)
    pass

plt.legend()
plt.xscale('log')
plt.xlabel(r'Energy (MeV)')
plt.ylabel(r'S (eV b)')
plt.yscale('log')

plt.title(r'$d$($p$,$\gamma$)$^3$He Gaussian Process Regression Fit')

plt.show()

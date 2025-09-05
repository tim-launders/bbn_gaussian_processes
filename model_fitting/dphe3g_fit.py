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
from scipy.special import kv

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

scaling = 1e6

E = np.concatenate(all_E)
logE = np.log(E)
S = scaling*np.concatenate(all_S)
total_error = scaling*np.concatenate(all_total_error)
systematic_error = scaling*np.concatenate(all_systematic_error)
statistical_error = scaling*np.concatenate(all_statistical_error)

# Load in theory curve

file = './data/dphe3g_theory_curve.csv'
theory_E, theory_S = np.loadtxt(file, delimiter=',', unpack=True)
sort_idx = np.argsort(theory_E)
x_sorted = np.log(theory_E[sort_idx])
y_sorted = scaling*theory_S[sort_idx]
theory_mean_base = UnivariateSpline(x=x_sorted, y=y_sorted, s=0)
theory_mean = lambda X: jnp.array([theory_mean_base(x) if x > x_sorted[0] else y_sorted[0] for x in X])
#theory_mean = lambda X : jnp.zeros_like(X)


hyper = [1.27620949E+07, 6.66993882E+00, 1.73922457E-02, 5.74876743E+02]

global_kernel = Constant(hyper[0], prior_params=[1e-13,1e10]) * SE(hyper[1], prior_params=[1e-6,1e8]) 
local_kernel = Constant(hyper[2], prior_params=[1e-16,1e6]) * Matern(hyper[3], nu=0.25, prior_params=[1e-6,1e14])
kernel = global_kernel + local_kernel

gpcov = GPR(kernel)
cov_mat = gpcov.calculate_covariance(total_error, systematic_error, points)

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
        scale_mat[row, col] += row_error * col_error
        # Repeat for symmetric part
        scale_mat[col, row] = scale_mat[row, col]
cov_mat += scale_mat


S_subtract = S - theory_mean(logE)

optimizer = LeaveDatasetOutOptimizer(kernel, logE, S_subtract, cov_mat, points=points)

print('Initial loss: ', optimizer.loss(jnp.log(kernel.get_params())))
"""
params, min_loss = optimizer.adam(tolerance=1e-5)
print('\nFinal loss: ', min_loss)
print('Optimized hyperparameter values: ', params)
kernel.set_params(params)
"""
gp = GPR(kernel, noise=0, mean_func=theory_mean)
gp.fit(logE, S, total_error, systematic_error, points)
gp.set_covariance(cov_mat)

num_points = 200

E_pred = jnp.logspace(np.log10(2.5e-3), np.log10(2), num_points)

mean, sigma = gp.predict(jnp.log(E_pred), True)

for i in range(len(points)):
    plt.errorbar(all_E[i], scaling*all_S[i], scaling*all_total_error[i], fmt=".", color=colors[i], label=sets[i])
plt.plot(E_pred, mean, color='black', label='GPR Mean')
plt.fill_between(E_pred, mean-sigma, mean+sigma, alpha=0.4, linewidth=0, color='black')

samples = gp.draw_samples(jnp.log(E_pred), 3, seed=5)
for sample in samples: 
    plt.plot(E_pred, sample)
    pass

plt.legend()
plt.xscale('log')
plt.xlabel(r'Energy (MeV)')
plt.ylabel(r'S (eV b)')
plt.yscale('log')

plt.title(r'$d$($p$,$\gamma$)$^3$He Gaussian Process Regression Fit')
plt.show()

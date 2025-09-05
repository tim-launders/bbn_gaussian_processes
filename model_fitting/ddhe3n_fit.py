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

data_dir = './data/ddHe3n/'

sets = ['BR90', 'GR95', 'SC72', 'Leo06', 'KR87B', 'KR87M']
colors = ['blue', 'cyan', 'brown', 'red', 'indigo', 'seagreen']

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
total_error = np.concatenate(all_total_error)
systematic_error = np.concatenate(all_systematic_error)
statistical_error = np.concatenate(all_statistical_error)
#total_error = statistical_error                            # Uncomment for only statistical error (also multiply systematic_error by 0)

# Load in theory curve

file = './data/ddhe3n_theory_curve.csv'
theory_E, theory_S = np.loadtxt(file, delimiter=',', unpack=True)
sort_idx = np.argsort(theory_E)
x_sorted = np.log(theory_E[sort_idx])
y_sorted = theory_S[sort_idx]
theory_mean = UnivariateSpline(x=x_sorted, y=y_sorted, s=0)
theory_mean = lambda x: 0

hyper = [2.43637577E+05, 3.43956298E+01, 4.60471861E-03, 2.17241095E+03]

global_kernel = Constant(hyper[0], prior_params=[1e-6,1e8]) * SE(hyper[1], prior_params=[1e-6,1e6]) 
local_kernel = Constant(hyper[2], prior_params=[1e-7, 1e7]) * Matern(hyper[3], nu=0.25, prior_params=[1e-6,1e8])
kernel = global_kernel + local_kernel

gpcov = GPR(kernel)
cov_mat = gpcov.calculate_covariance(total_error, systematic_error, points)
S_subtract = S - theory_mean(logE)

optimizer = LeaveDatasetOutOptimizer(kernel, logE, S_subtract, cov_mat, points=points)

print('Initial loss: ', optimizer.loss(jnp.log(kernel.get_params())))
"""
params, min_loss = optimizer.adam(tolerance=1e-5)
print('\nFinal loss: ', min_loss)
print('Optimized hyperparameter values: ', params)
kernel.set_params(params)
"""
gp = GPR(kernel, mean_func=theory_mean)
gp.fit(logE, S, total_error, systematic_error, points)

num_points = 200

E_pred = jnp.logspace(np.log10(5e-3), np.log10(3.2), num_points)

mean, sigma = gp.predict(jnp.log(E_pred), True)

for i in range(len(points)):
    plt.errorbar(all_E[i], all_S[i], all_systematic_error[i], fmt=".", color=colors[i], label=sets[i])
plt.plot(E_pred, mean, color='black', label='GPR Mean')
plt.fill_between(E_pred, mean-sigma, mean+sigma, alpha=0.4, linewidth=0, color='black')

samples = gp.draw_samples(jnp.log(E_pred), 3, seed=9)
for sample in samples: 
    plt.plot(E_pred, sample)
    pass

plt.legend()
plt.xscale('log')
plt.xlabel(r'Energy (MeV)')
plt.ylabel(r'S (MeV b)')

plt.title(r'$d$($d$,$n$)$^3$He Gaussian Process Regression Fit')

#plt.yscale('log')
plt.show()

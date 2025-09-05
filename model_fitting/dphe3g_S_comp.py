# This file compares directly the S-factors predicted by the GP, PRIMAT, and PArthENoPE


import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import jax
from scipy.interpolate import UnivariateSpline
from params import parth_S_dpHe3g
from gaussian_processes.kernels import Constant, SE, Dot, Matern, RationalQuadratic
from gaussian_processes.gp import GPR

# ddHe3n data

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
total_error = np.concatenate(all_total_error)
systematic_error = np.concatenate(all_systematic_error)
statistical_error = np.concatenate(all_statistical_error)

def lognormal_var(error, mean):
    return np.sqrt(np.log((error/mean)**2+1))

log_total_error = lognormal_var(total_error, S)
log_systematic_error = lognormal_var(systematic_error, S)
log_statistical_error = lognormal_var(statistical_error, S)

gpcov = GPR(kernel=SE(1.0))
cov_mat = gpcov.calculate_covariance(1e6*total_error, 1e6*systematic_error, points)
scale_mat = np.zeros_like(cov_mat)
start_ind = np.sum(points[0:-2])
scale_factor = 0.12
for i in range(points[-2]):
    for j in range(points[-1]):
        row = start_ind + i
        col = start_ind + points[-2] + j
        S_row = 1e6*S[row]
        S_col = 1e6*S[col]
        row_error = scale_factor * S_row
        col_error = scale_factor * S_col
        scale_mat[row, col] += row_error * col_error
        # Repeat for symmetric part
        scale_mat[col, row] = scale_mat[row, col]
cov_mat += scale_mat

log_cov_mat = gpcov.calculate_covariance(log_total_error, log_systematic_error, points)
log_scale_mat = np.zeros_like(cov_mat)
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
        log_scale_mat[row, col] += log_row_error * log_col_error
        # Repeat for symmetric part
        log_scale_mat[col, row] = log_scale_mat[row, col]
log_cov_mat += log_scale_mat

hyper = [1.27620949E+07, 6.66993882E+00, 1.73922457E-02, 5.74876743E+02]

global_kernel = Constant(hyper[0], prior_params=[1e-6,1e6]) * SE(hyper[1], prior_params=[1e-6,1e6]) 
local_kernel = Constant(hyper[2], prior_params=[1e-7, 1e7]) * Matern(hyper[3], nu=0.25, prior_params=[1e-6,1e6])
kernel = global_kernel + local_kernel

gp = GPR(kernel, noise=0)
gp.fit(logE, 1e6*S, 1e6*total_error, 1e6*systematic_error, points)
gp.set_covariance(cov_mat)

E_plot = np.logspace(np.log10(2.5e-3), np.log10(2), 300)
gp_S, gp_S_sigma = gp.predict(np.log(E_plot), True)
gp_S = 1e-6*gp_S

# Add log GP

log_hyper = [2.31789296E+03, 7.60324923E+00, 2.66962082E-02, 3.74584365E+02]
log_kernel = Constant(log_hyper[0]) * SE(log_hyper[1]) + Constant(log_hyper[2]) * Matern(log_hyper[3], nu=0.25)
log_gp = GPR(log_kernel, noise=0)
log_gp.fit(logE, np.log(S), log_total_error, log_systematic_error, points)
log_gp.set_covariance(log_cov_mat)
log_gp_S = log_gp.predict(np.log(E_plot))
log_gp_S = np.exp(log_gp_S)

# Add theory GP 

file = './data/dphe3g_theory_curve.csv'
theory_E, theory_S = np.loadtxt(file, delimiter=',', unpack=True)
sort_idx = np.argsort(theory_E)
x_sorted = np.log(theory_E[sort_idx])
y_sorted = 1e6*theory_S[sort_idx]
theory_mean_base = UnivariateSpline(x=x_sorted, y=y_sorted, s=0)
theory_mean = lambda X: np.array([theory_mean_base(x) if x > x_sorted[0] else y_sorted[0] for x in X])

theory_hyper = [2.93908576E+06, 1.76474835E+01, 1.54271399E-02, 6.48090944E+02]
theory_kernel = Constant(theory_hyper[0]) * SE(theory_hyper[1]) + Constant(theory_hyper[2]) * Matern(theory_hyper[3], nu=0.25)
theory_gp = GPR(theory_kernel, mean_func=theory_mean)
theory_gp.fit(logE, 1e6*S, 1e6*total_error, 1e6*systematic_error, points)
theory_gp.set_covariance(cov_mat)
theory_gp_S = theory_gp.predict(np.log(E_plot))
theory_gp_S = 1e-6*theory_gp_S

ddhe3n_primat_file = './data/primat_dphe3g_S.csv'
primat_E, primat_S = np.loadtxt(ddhe3n_primat_file, delimiter=',', unpack=True)

parth_S = parth_S_dpHe3g(E_plot)

plt.figure(figsize=(8,6))

for i in range(len(points)):
    gp_comp = gp.predict(np.log(all_E[i]))
    gp_comp = 1e-6*gp_comp
    plt.errorbar(all_E[i], all_S[i]/gp_comp, all_total_error[i]/gp_comp, fmt=".", color=colors[i])

gp_comp_primat = gp.predict(np.log(primat_E))
gp_comp_primat = 1e-6*gp_comp_primat

plt.plot(primat_E, primat_S/gp_comp_primat, color='red', label='PRIMAT Mean')
plt.plot(E_plot, parth_S/gp_S, color='blue', label='PArthENoPE Mean')
plt.plot(E_plot, theory_gp_S/gp_S, color='purple', label='Theory GP Mean')
plt.plot(E_plot, log_gp_S/gp_S, color='seagreen', label='Log GP Mean')
plt.plot(E_plot, [1 for i in range(len(E_plot))], color='black', linestyle='dashed', label='Zero GP Mean')

plt.xscale('log')
plt.legend(loc='upper left')
plt.xlabel('Energy (MeV)')
plt.ylabel('S / GP Mean S')
plt.title(r'$d$($p$,$\gamma$)$^3$He Normalized S-Factor')
plt.tight_layout()
plt.show()

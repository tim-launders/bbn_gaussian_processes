# This file compares directly the S-factors predicted by the GP, PRIMAT, and PArthENoPE


import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import jax
from scipy.interpolate import UnivariateSpline
from params import parth_S_ddtp
from gaussian_processes.kernels import Constant, SE, Dot, Matern, RationalQuadratic
from gaussian_processes.gp import GPR

# ddHe3n data

jax.config.update('jax_enable_x64', True)

data_dir = './data/ddtp/'

sets = ['KR87B', 'KR87M', 'BR90', 'GR95', 'GR95C', 'Leo06']
colors = ['indigo', 'seagreen', 'blue', 'cyan', 'brown', 'red']

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

hyper = [4.24454682E+03, 1.38349176E+01, 9.35852483E-04, 1.06890577E+03]

global_kernel = Constant(hyper[0], prior_params=[1e-6,1e6]) * SE(hyper[1], prior_params=[1e-6,1e6]) 
local_kernel = Constant(hyper[2], prior_params=[1e-7, 1e7]) * Matern(hyper[3], nu=0.25, prior_params=[1e-6,1e6])
kernel = global_kernel + local_kernel

gp = GPR(kernel)
gp.fit(logE, S, total_error, systematic_error, points)

E_plot = np.logspace(np.log10(1e-3), np.log10(1), 200)
gp_S, gp_S_sigma = gp.predict(np.log(E_plot), True)

# Add theory GP and uncorrelated GP 

theory_file = 'data/ddtp_theory_curve.csv'
theory_E, theory_S = np.loadtxt(theory_file, delimiter=',', unpack=True)
sort_idx = np.argsort(theory_E)
x_sorted = np.log(theory_E[sort_idx])
y_sorted = theory_S[sort_idx]
theory_mean = UnivariateSpline(x=x_sorted, y=y_sorted, s=0)

theory_hyper = [9.40732505E+01, 2.44754447E+02, 9.25935353E-04, 1.07982256E+03]
theory_kernel = Constant(theory_hyper[0]) * SE(theory_hyper[1]) + Constant(theory_hyper[2]) * Matern(theory_hyper[3], nu=0.25)
theory_gp = GPR(theory_kernel, mean_func=theory_mean)
theory_gp.fit(logE, S, total_error, systematic_error, points)
theory_gp_S = theory_gp.predict(np.log(E_plot))

uncorrelated_hyper = [9.61960754E+04, 2.30694326E+01, 5.90475607E-05, 1.69410532E+04]
uncorrelated_kernel = Constant(uncorrelated_hyper[0]) * SE(uncorrelated_hyper[1]) + Constant(uncorrelated_hyper[2]) * Matern(uncorrelated_hyper[3], nu=0.25)
uncorrelated_gp = GPR(uncorrelated_kernel)
uncorrelated_gp.fit(logE, S, total_error, 0*systematic_error, points)
uncorrelated_gp_S = uncorrelated_gp.predict(np.log(E_plot))

mle_hyper = [3.33386014E-02, 3.10473244E+00, 6.54885896E-04, 1.63510612E+03]
mle_kernel = Constant(mle_hyper[0]) * SE(mle_hyper[1]) + Constant(mle_hyper[2]) * Matern(mle_hyper[3], nu=0.25)
mle_gp = GPR(mle_kernel)
mle_gp.fit(logE, S, total_error, systematic_error, points)
mle_gp_S = mle_gp.predict(np.log(E_plot))

stat_hyper = [1.94997463E+02, 8.86171592E+00, 2.52202275E-03, 4.20807274E+02]
stat_kernel = Constant(stat_hyper[0]) * SE(stat_hyper[1]) + Constant(stat_hyper[2]) * Matern(stat_hyper[3], nu=0.25)
stat_gp = GPR(stat_kernel)
stat_gp.fit(logE, S, statistical_error, 0*systematic_error, points)
stat_gp_S = stat_gp.predict(np.log(E_plot))

"""
no95c_hyper = [2.33828998E+03, 2.40364719E+01, 1.23402458E-03, 8.10630538E+02]
no95c_kernel = Constant(no95c_hyper[0]) * SE(no95c_hyper[1]) + Constant(no95c_hyper[2]) * Matern(no95c_hyper[3], nu=0.25)
no95c_gp = GPR(no95c_kernel)
all_E.pop(-2)
all_S.pop(-2)
all_total_error.pop(-2)
all_systematic_error.pop(-2)
points.pop(-2)

no95c_logE = np.log(np.concatenate(all_E))
no95c_S = np.concatenate(all_S)
no95c_total_error = np.concatenate(all_total_error)
no95c_systematic_error = np.concatenate(all_systematic_error)
no95c_gp.fit(no95c_logE, no95c_S, no95c_total_error, no95c_systematic_error, points)
no95c_gp_S = no95c_gp.predict(np.log(E_plot))
"""

ddtp_primat_file = './data/primat_ddtp_S.csv'
primat_E, primat_S = np.loadtxt(ddtp_primat_file, delimiter=',', unpack=True)

parth_S = parth_S_ddtp(E_plot)

fig, ax = plt.subplots(figsize=(11,7.5))

for i in range(len(points)):
    gp_comp = gp.predict(np.log(all_E[i]))
    ax.errorbar(all_E[i], all_S[i]/gp_comp, all_total_error[i]/gp_comp, fmt=".", color=colors[i])

gp_comp_primat = gp.predict(np.log(primat_E))

ax.plot(primat_E, primat_S/gp_comp_primat, color='red', label='PRIMAT Mean')
ax.plot(E_plot, parth_S/gp_S, color='blue', label='PArthENoPE Mean')
ax.plot(E_plot, theory_gp_S/gp_S, color='purple', label='Theory GP Mean')
ax.plot(E_plot, uncorrelated_gp_S/gp_S, color='seagreen', label='Uncorrelated GP Mean')
ax.plot(E_plot, mle_gp_S/gp_S, color='brown', label='Marginal Likelihood GP Mean')
#ax.plot(E_plot, stat_gp_S/gp_S, color='grey', label='Statistical Only')
ax.plot(E_plot, [1 for i in range(len(E_plot))], color='black', linestyle='dashed', label='Zero GP Mean')

ax.set_xscale('log')
ax.legend(loc='upper left')
ax.set_xlabel('Energy (MeV)')
ax.set_ylabel('S / GP Mean S')
ax.set_ylim(0.6, 1.6)

axins = ax.inset_axes([0.575, 0.675, 0.4, 0.3])

for i in range(len(points)):
    gp_comp = gp.predict(np.log(all_E[i]))
    axins.errorbar(all_E[i], all_S[i]/gp_comp, all_total_error[i]/gp_comp, fmt=".", color=colors[i])

axins.plot(E_plot, [1 for i in range(len(E_plot))], color='black', linestyle='dashed')
axins.plot(primat_E, primat_S/gp_comp_primat, color='red')
axins.plot(E_plot, parth_S/gp_S, color='blue')
axins.plot(E_plot, theory_gp_S/gp_S, color='purple')
axins.plot(E_plot, uncorrelated_gp_S/gp_S, color='seagreen')
axins.plot(E_plot, mle_gp_S/gp_S, color='brown')
axins.set_xlim(0.018, 0.1)
axins.set_ylim(0.93, 1.13)
axins.set_xscale('log')
axins.set_yticks([])
axins.set_xticks([])
axins.tick_params(bottom=False, top=False, labelbottom=False)
axins.minorticks_off()

ind = ax.indicate_inset_zoom(axins, edgecolor='black', facecolor='none', alpha=1.0)
for conn in ind.connectors:
    conn.set_visible(False)
ind.rectangle.set_edgecolor('black')
ind.rectangle.set_linewidth(1.0)
ind.rectangle.set_facecolor('none')

ax.set_title(r'$d$($d$,$p$)$t$ Normalized S-Factor')

plt.tight_layout()
plt.show()
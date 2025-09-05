# This file computes the D/H posterior from sampling from the GP posterior for S-factors for the 3 crucial deuterium reaactions. 


import sys
sys.path.append('.')

import numpy as np
import jax.numpy as jnp
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from sample_rates import dd_R, dp_R
from gaussian_processes.kernels import Constant, SE, Dot, Matern
from gaussian_processes.gp import GPR
from LINX.linx.data.nuclear_rates.key_PArthENoPE import ddHe3n_frwrd_rate, ddtp_frwrd_rate, dpHe3g_frwrd_rate
from LINX.linx.abundances import AbundanceModel
from LINX.linx.background import BackgroundModel
from LINX.linx.nuclear import NuclearRates
from LINX.linx.thermo import T_g

data_dir = './data/'


# Load in d+d->He3+n data, fit a GP, and draw sample rates


ddhe3n_sets = ['BR90', 'GR95', 'SC72', 'Leo06', 'KR87B', 'KR87M']

ddhe3n_all_E = []
ddhe3n_all_S = []
ddhe3n_all_total_error = []
ddhe3n_all_systematic_error = []
ddhe3n_points = []

for set in ddhe3n_sets:
    energy, cross, S, total, sys = np.loadtxt(data_dir+'ddHe3n/'+set+'.txt', unpack=True)
    ddhe3n_all_E.append(energy)
    ddhe3n_all_S.append(S)
    ddhe3n_all_total_error.append(total)
    ddhe3n_all_systematic_error.append(sys)
    ddhe3n_points.append(len(S))

ddhe3n_E = np.concatenate(ddhe3n_all_E)
ddhe3n_logE = np.log(ddhe3n_E)
ddhe3n_S = np.concatenate(ddhe3n_all_S)
ddhe3n_total_error = np.concatenate(ddhe3n_all_total_error)
ddhe3n_systematic_error = np.concatenate(ddhe3n_all_systematic_error)

# Import theory curve for GP mean

ddhe3n_file = './data/ddhe3n_theory_curve.csv'
ddhe3n_theory_E, ddhe3n_theory_S = np.loadtxt(ddhe3n_file, delimiter=',', unpack=True)
ddhe3n_sort_idx = np.argsort(ddhe3n_theory_E)
ddhe3n_x_sorted = np.log(ddhe3n_theory_E[ddhe3n_sort_idx])
ddhe3n_y_sorted = ddhe3n_theory_S[ddhe3n_sort_idx]
ddhe3n_theory_mean = UnivariateSpline(x=ddhe3n_x_sorted, y=ddhe3n_y_sorted, s=0)
ddhe3n_theory_mean = lambda x: 0                                       # Uncomment when using zero mean prior

# Load in parameters for the kernel and fit the GP

#ddhe3n_hyper = [4.95230230E+04, 1.63550757E+01, 4.65902731E-03, 2.14710930E+03]         # Theory mean
ddhe3n_hyper = [2.43637577E+05, 3.43956298E+01, 4.60471861E-03, 2.17241095E+03]         # Zero mean
#ddhe3n_hyper = [3.25565503E+06, 7.05118447E+01, 4.17345785E-06, 2.39691819E+06]         # No correlations, zero mean
#ddhe3n_hyper = [9.90368587E-01, 4.91453150E+00, 3.05801581E-03, 3.28278797E+03]         # MLE, zero mean
ddhe3n_kernel = Constant(ddhe3n_hyper[0]) * SE(ddhe3n_hyper[1]) + Constant(ddhe3n_hyper[2]) * Matern(ddhe3n_hyper[3], nu=0.25)

ddhe3n_gp = GPR(ddhe3n_kernel, mean_func=ddhe3n_theory_mean)
ddhe3n_gp.fit(ddhe3n_logE, ddhe3n_S, ddhe3n_total_error, ddhe3n_systematic_error, ddhe3n_points)

# Draw samples

num_samples = 10000
num_points = 1000
ddhe3n_E_pred = jnp.logspace(jnp.log10(5e-3), jnp.log10(3.2), num_points)
ddhe3n_S_mean = ddhe3n_gp.predict(jnp.log(ddhe3n_E_pred))
ddhe3n_S_samples = ddhe3n_gp.draw_samples(jnp.log(ddhe3n_E_pred), num_samples)

# Convert S-factor samples to rates

ddhe3n_primat_file = './LINX/linx/data/nuclear_rates/key_PRIMAT_2023/ddHe3n.txt'
T9, ddhe3n_r_primat, ddhe3n_f = np.loadtxt(ddhe3n_primat_file, unpack=True)
T_primat = T9 * 1e9
first = 17          # Indices for rate comparison
last = 51
num_T = len(T_primat)

ddhe3n_R_samples = []
for sample in ddhe3n_S_samples:
    rate = jnp.array([dd_R(t, ddhe3n_E_pred, sample) for t in T_primat])
    ddhe3n_R_samples.append(rate)
ddhe3n_R_samples = jnp.array(ddhe3n_R_samples)

# Calculate mean and 1 sigma ad each temperature

ddhe3n_gp_R_mean = jnp.mean(ddhe3n_R_samples, axis=0)
ddhe3n_gp_R_sigma = jnp.std(ddhe3n_R_samples, axis=0)

# Plot rate posterior with 1 sigma envelope

figure_dir = './abundance_calculations/figures/deuterium/'

T_plot = T_primat[first:last]
ddhe3n_gp_R_mean_plot = ddhe3n_gp_R_mean[first:last]
ddhe3n_gp_R_sigma_plot = ddhe3n_gp_R_sigma[first:last]
ddhe3n_gp_sigma_ratio = ddhe3n_gp_R_sigma_plot / ddhe3n_gp_R_mean_plot

ddhe3n_primat_up = np.multiply(ddhe3n_r_primat, ddhe3n_f)[first:last]
ddhe3n_primat_down = np.multiply(ddhe3n_r_primat, 2-ddhe3n_f)[first:last]
ddhe3n_primat_ratio_up = ddhe3n_primat_up / ddhe3n_gp_R_mean_plot
ddhe3n_primat_ratio_down = ddhe3n_primat_down / ddhe3n_gp_R_mean_plot

ddhe3n_parth_up = np.array([ddHe3n_frwrd_rate(t, 1) for t in T_plot])
ddhe3n_parth_down = np.array([ddHe3n_frwrd_rate(t, -1) for t in T_plot])
ddhe3n_parth_ratio_up = ddhe3n_parth_up / ddhe3n_gp_R_mean_plot
ddhe3n_parth_ratio_down = ddhe3n_parth_down / ddhe3n_gp_R_mean_plot

#plt.plot(T_plot, 1 - ddhe3n_gp_sigma_ratio, color='black', label=r'GP $1\sigma$ envelope')        # Uncomment for lines to outline envelope
#plt.plot(T_plot, 1 + ddhe3n_gp_sigma_ratio, color='black')
plt.fill_between(T_plot, 1-ddhe3n_gp_sigma_ratio, 1+ddhe3n_gp_sigma_ratio, color='black', alpha=0.3, linewidth=0, label=r'GP $1\sigma$ envelope')
plt.fill_between(T_plot, ddhe3n_primat_ratio_down, ddhe3n_primat_ratio_up, color='red', alpha=0.5, linewidth=0, label=r'PRIMAT $1\sigma$')
plt.fill_between(T_plot, ddhe3n_parth_ratio_down, ddhe3n_parth_ratio_up, color='blue', alpha=0.5, linewidth=0, label=r'PArthENoPE $1\sigma$')
for i in range(5):
    sample = ddhe3n_R_samples[i][first:last]
    plt.plot(T_plot, sample/ddhe3n_gp_R_mean_plot)
plt.legend()
plt.xscale('log')
plt.xlabel('T (K)')
plt.ylabel('Rate / GP mean rate')
plt.title(f'ddHe3n GP Rate Posterior, {num_samples} samples')
plt.savefig(figure_dir+'ddhe3n_rate_envelope.png')
plt.clf()


# Output mean and 1 sigma GP rates as a txt file


out_dir = './data/output/deuterium/'

with open(out_dir+'ddhe3n_gp_rates.txt', 'w') as f:
    f.write('T (K)\tMean Rate\t1 sigma')
    for i in range(num_T):
        f.write('\n'+str(T_primat[i])+'\t'+str(ddhe3n_gp_R_mean[i])+'\t'+str(ddhe3n_gp_R_sigma[i]))



# Load in d+d->H3+p data, fit a GP, and get rate samples



ddtp_sets = ['KR87B', 'KR87M', 'BR90', 'GR95', 'GR95C', 'Leo06']

ddtp_all_E = []
ddtp_all_S = []
ddtp_all_total_error = []
ddtp_all_systematic_error = []
ddtp_points = []

for set in ddtp_sets:
    energy, cross, S, total, sys = np.loadtxt(data_dir+'ddtp/'+set+'.txt', unpack=True)
    ddtp_all_E.append(energy)
    ddtp_all_S.append(S)
    ddtp_all_total_error.append(total)
    ddtp_all_systematic_error.append(sys)
    ddtp_points.append(len(S))

ddtp_E = np.concatenate(ddtp_all_E)
ddtp_logE = np.log(ddtp_E)
ddtp_S = np.concatenate(ddtp_all_S)
ddtp_total_error = np.concatenate(ddtp_all_total_error)
ddtp_systematic_error = np.concatenate(ddtp_all_systematic_error)

# Import theory curve for GP mean

ddtp_file = './data/ddtp_theory_curve.csv'
ddtp_theory_E, ddtp_theory_S = np.loadtxt(ddtp_file, delimiter=',', unpack=True)
ddtp_sort_idx = np.argsort(ddtp_theory_E)
ddtp_x_sorted = np.log(ddtp_theory_E[ddtp_sort_idx])
ddtp_y_sorted = ddtp_theory_S[ddtp_sort_idx]
ddtp_theory_mean = UnivariateSpline(x=ddtp_x_sorted, y=ddtp_y_sorted, s=0)
ddtp_theory_mean = lambda x: 0

# Load in parameters for the kernel and fit the GP

#ddtp_hyper = [9.40732505E+01, 2.44754447E+02, 9.25935353E-04, 1.07982256E+03]          # Theory mean
ddtp_hyper = [4.24454682E+03, 1.38349176E+01, 9.35852483E-04, 1.06890577E+03]          # Zero mean
#ddtp_hyper = [9.61960754E+04, 2.30694326E+01, 5.90475607E-05, 1.69410532E+04]          # No correlations, zero mean
#ddtp_hyper = [3.33386014E-02, 3.10473244E+00, 6.54885896E-04, 1.63510612E+03]          # MLE, zero mean
ddtp_kernel = Constant(ddtp_hyper[0]) * SE(ddtp_hyper[1]) + Constant(ddtp_hyper[2]) * Matern(ddtp_hyper[3], nu=0.25)

ddtp_gp = GPR(ddtp_kernel, mean_func=ddtp_theory_mean)
ddtp_gp.fit(ddtp_logE, ddtp_S, ddtp_total_error, ddtp_systematic_error, ddtp_points)

# Draw samples

ddtp_E_pred = jnp.logspace(np.log10(1e-3), np.log10(1), num_points)
ddtp_S_mean = ddtp_gp.predict(jnp.log(ddtp_E_pred))
ddtp_S_samples = ddtp_gp.draw_samples(jnp.log(ddtp_E_pred), num_samples)

# Convert S-factor samples to rates

ddtp_primat_file = './LINX/linx/data/nuclear_rates/key_PRIMAT_2023/ddtp.txt'
T9, ddtp_r_primat, ddtp_f = np.loadtxt(ddtp_primat_file, unpack=True)

ddtp_R_samples = []
for sample in ddtp_S_samples:
    rate = jnp.array([dd_R(t, ddtp_E_pred, sample) for t in T_primat])
    ddtp_R_samples.append(rate)
ddtp_R_samples = jnp.array(ddtp_R_samples)

# Calculate mean and 1 sigma ad each temperature

ddtp_gp_R_mean = jnp.mean(ddtp_R_samples, axis=0)
ddtp_gp_R_sigma = jnp.std(ddtp_R_samples, axis=0)

# Plot rate posterior with 1 sigma envelope

ddtp_gp_R_mean_plot = ddtp_gp_R_mean[first:last]
ddtp_gp_R_sigma_plot = ddtp_gp_R_sigma[first:last]
ddtp_gp_sigma_ratio = ddtp_gp_R_sigma_plot / ddtp_gp_R_mean_plot

ddtp_primat_up = np.multiply(ddtp_r_primat, ddtp_f)[first:last]
ddtp_primat_down = np.multiply(ddtp_r_primat, 2-ddtp_f)[first:last]
ddtp_primat_ratio_up = ddtp_primat_up / ddtp_gp_R_mean_plot
ddtp_primat_ratio_down = ddtp_primat_down / ddtp_gp_R_mean_plot

ddtp_parth_up = np.array([ddtp_frwrd_rate(t, 1) for t in T_plot])
ddtp_parth_down = np.array([ddtp_frwrd_rate(t, -1) for t in T_plot])
ddtp_parth_ratio_up = ddtp_parth_up / ddtp_gp_R_mean_plot
ddtp_parth_ratio_down = ddtp_parth_down / ddtp_gp_R_mean_plot

#plt.plot(T_plot, 1 - ddtp_gp_sigma_ratio, color='black', label=r'GP $1\sigma$ envelope')
#plt.plot(T_plot, 1 + ddtp_gp_sigma_ratio, color='black')
plt.fill_between(T_plot, 1-ddtp_gp_sigma_ratio, 1+ddtp_gp_sigma_ratio, color='black', alpha=0.3, linewidth=0, label=r'GP $1\sigma$ envelope')
plt.fill_between(T_plot, ddtp_primat_ratio_down, ddtp_primat_ratio_up, color='red', alpha=0.5, linewidth=0, label=r'PRIMAT $1\sigma$')
plt.fill_between(T_plot, ddtp_parth_ratio_down, ddtp_parth_ratio_up, color='blue', alpha=0.5, linewidth=0, label=r'PArthENoPE $1\sigma$')
for i in range(5):
    sample = ddtp_R_samples[i][first:last]
    plt.plot(T_plot, sample/ddtp_gp_R_mean_plot)
plt.legend()
plt.xscale('log')
plt.xlabel('T (K)')
plt.ylabel('Rate / GP mean rate')
plt.title(f'ddtp GP Rate Posterior, {num_samples} samples')
plt.savefig(figure_dir+'ddtp_rate_envelope.png')
plt.clf()

# Output mean and 1 sigma GP rates as a txt file

with open(out_dir+'ddtp_gp_rates.txt', 'w') as f:
    f.write('T (K)\tMean Rate\t1 sigma')
    for i in range(num_T):
        f.write('\n'+str(T_primat[i])+'\t'+str(ddtp_gp_R_mean[i])+'\t'+str(ddtp_gp_R_sigma[i]))



# Load in d+p->He3+g data and fit a GP, fit in eV b instead of MeV b



dphe3g_sets = ['GR62', 'GR63', 'WA63', 'MA97', 'SC97', 'CA02', 'TI19', 'MO20', 'TU21 3', 'TU21 4']

dphe3g_all_E = []
dphe3g_all_S = []
dphe3g_all_total_error = []
dphe3g_all_systematic_error = []
dphe3g_all_statistical_error = []
dphe3g_points = []

for set in dphe3g_sets:
    energy, cross, Sfactor, total, sys = np.loadtxt(data_dir+'dpHe3g/'+set+'.txt', unpack=True)
    dphe3g_all_E.append(energy)
    dphe3g_all_S.append(Sfactor)
    dphe3g_all_total_error.append(total)
    dphe3g_all_systematic_error.append(sys)
    dphe3g_all_statistical_error.append(np.sqrt(total**2 - sys**2))
    dphe3g_points.append(len(Sfactor))

dphe3g_E = np.concatenate(dphe3g_all_E)
dphe3g_logE = np.log(dphe3g_E)
dphe3g_S = np.concatenate(dphe3g_all_S)
dphe3g_log_S = np.log(dphe3g_S)
dphe3g_total_error = np.concatenate(dphe3g_all_total_error)
dphe3g_systematic_error = np.concatenate(dphe3g_all_systematic_error)
dphe3g_statistical_error = np.concatenate(dphe3g_all_statistical_error)

def lognormal_var(error, mean):
    return np.sqrt(np.log((error/mean)**2+1))
    
log_dphe3g_total_error = lognormal_var(dphe3g_total_error, dphe3g_S)
log_dphe3g_systematic_error = lognormal_var(dphe3g_systematic_error, dphe3g_S)
log_dphe3g_statistical_error = lognormal_var(dphe3g_statistical_error, dphe3g_S)

# Load in theory curve

dphe3g_file = './data/dphe3g_theory_curve.csv'
dphe3g_theory_E, dphe3g_theory_S = np.loadtxt(dphe3g_file, delimiter=',', unpack=True)
dphe3g_sort_idx = np.argsort(dphe3g_theory_E)
dphe3g_x_sorted = np.log(dphe3g_theory_E[dphe3g_sort_idx])
dphe3g_y_sorted = dphe3g_theory_S[dphe3g_sort_idx]*1e6
dphe3g_theory_mean_base = UnivariateSpline(x=dphe3g_x_sorted, y=dphe3g_y_sorted, s=0)
dphe3g_theory_mean = lambda X: jnp.array([dphe3g_theory_mean_base(x) if x > dphe3g_x_sorted[0] else dphe3g_y_sorted[0] for x in X])
dphe3g_theory_mean = lambda X: 0

# Load in parameters for the kernel and fit the GP

dphe3g_hyper = [2.31789296E+03, 7.60324923E+00, 2.66962082E-02, 3.74584365E+02]
dphe3g_kernel = Constant(dphe3g_hyper[0]) * SE(dphe3g_hyper[1]) + Constant(dphe3g_hyper[2]) * Matern(dphe3g_hyper[3], nu=0.25)

dphe3g_gp = GPR(dphe3g_kernel, mean_func=dphe3g_theory_mean)
dphe3g_gp.fit(dphe3g_logE, dphe3g_log_S, log_dphe3g_total_error, log_dphe3g_systematic_error, dphe3g_points)

# Add dependencies between TU21 datasets

cov_mat = dphe3g_gp.calculate_covariance(log_dphe3g_total_error, log_dphe3g_systematic_error, dphe3g_points)
scale_mat = np.zeros_like(cov_mat)
start_ind = np.sum(dphe3g_points[0:-2])
scale_factor = 0.12
for i in range(dphe3g_points[-2]):
    for j in range(dphe3g_points[-1]):
        row = start_ind + i
        col = start_ind + dphe3g_points[-2] + j
        S_row = dphe3g_S[row]
        S_col = dphe3g_S[col]
        row_error = scale_factor * S_row
        col_error = scale_factor * S_col
        log_row_error = lognormal_var(row_error, S_row)
        log_col_error = lognormal_var(col_error, S_col)
        scale_mat[row, col] += log_row_error * log_col_error
        # Repeat for symmetric part
        scale_mat[col, row] = scale_mat[row, col]
cov_mat += scale_mat
dphe3g_gp.set_covariance(cov_mat)

# Draw samples

dphe3g_E_pred = jnp.logspace(np.log10(2.5e-3), np.log10(2), num_points)
dphe3g_S_mean = dphe3g_gp.predict(jnp.log(dphe3g_E_pred))
dphe3g_S_samples = dphe3g_gp.draw_samples(jnp.log(dphe3g_E_pred), num_samples)
dphe3g_S_samples = jnp.exp(dphe3g_S_samples)

# Convert S-factor samples to rates

dphe3g_primat_file = './LINX/linx/data/nuclear_rates/key_PRIMAT_2023/dpHe3g.txt'
T9, dphe3g_r_primat, dphe3g_f = np.loadtxt(dphe3g_primat_file, unpack=True)

dphe3g_R_samples = []
for sample in dphe3g_S_samples:
    rate = jnp.array([dp_R(t, dphe3g_E_pred, sample) for t in T_primat])
    dphe3g_R_samples.append(rate)
dphe3g_R_samples = jnp.array(dphe3g_R_samples)

# Calculate mean and 1 sigma ad each temperature

dphe3g_gp_R_mean = jnp.mean(dphe3g_R_samples, axis=0)
dphe3g_gp_R_sigma = jnp.std(dphe3g_R_samples, axis=0)

# Plot rate posterior with 1 sigma envelope

dphe3g_gp_R_mean_plot = dphe3g_gp_R_mean[first:last]
dphe3g_gp_R_sigma_plot = dphe3g_gp_R_sigma[first:last]
dphe3g_gp_sigma_ratio = dphe3g_gp_R_sigma_plot / dphe3g_gp_R_mean_plot

dphe3g_primat_up = np.multiply(dphe3g_r_primat, dphe3g_f)[first:last]
dphe3g_primat_down = np.multiply(dphe3g_r_primat, 2-dphe3g_f)[first:last]
dphe3g_primat_ratio_up = dphe3g_primat_up / dphe3g_gp_R_mean_plot
dphe3g_primat_ratio_down = dphe3g_primat_down / dphe3g_gp_R_mean_plot

dphe3g_parth_up = np.array([dpHe3g_frwrd_rate(t, 1) for t in T_plot])
dphe3g_parth_down = np.array([dpHe3g_frwrd_rate(t, -1) for t in T_plot])
dphe3g_parth_ratio_up = dphe3g_parth_up / dphe3g_gp_R_mean_plot
dphe3g_parth_ratio_down = dphe3g_parth_down / dphe3g_gp_R_mean_plot

#plt.plot(T_plot, 1 - dphe3g_gp_sigma_ratio, color='black', label=r'GP $1\sigma$ envelope')
#plt.plot(T_plot, 1 + dphe3g_gp_sigma_ratio, color='black')
plt.fill_between(T_plot, 1-dphe3g_gp_sigma_ratio, 1+dphe3g_gp_sigma_ratio, color='black', alpha=0.3, linewidth=0, label=r'GP $1\sigma$ envelope')
plt.fill_between(T_plot, dphe3g_primat_ratio_down, dphe3g_primat_ratio_up, color='red', alpha=0.5, linewidth=0, label=r'PRIMAT $1\sigma$')
plt.fill_between(T_plot, dphe3g_parth_ratio_down, dphe3g_parth_ratio_up, color='blue', alpha=0.5, linewidth=0, label=r'PArthENoPE $1\sigma$')
for i in range(5):
    sample = dphe3g_R_samples[i][first:last]
    plt.plot(T_plot, sample/dphe3g_gp_R_mean_plot)
plt.legend()
plt.xscale('log')
plt.xlabel('T (K)')
plt.ylabel('Rate / GP mean rate')
plt.title(f'dpHe3g GP Rate Posterior, {num_samples} samples')
plt.savefig(figure_dir+'dphe3g_rate_envelope.png')
plt.clf()

# Output mean and 1 sigma GP rates as a txt file

with open(out_dir+'dphe3g_gp_rates.txt', 'w') as f:
    f.write('T (K)\tMean Rate\t1 sigma')
    for i in range(num_T):
        f.write('\n'+str(T_primat[i])+'\t'+str(dphe3g_gp_R_mean[i])+'\t'+str(dphe3g_gp_R_sigma[i]))



# Set up nuclear net for LINX



bkg_model = BackgroundModel()
t_vec, a_vec, rho_g_vec, rho_nu_vec, rho_NP_vec, p_NP_vec, Neff_vec = bkg_model(jnp.asarray(0.))
T_g_vec = T_g(rho_g_vec)

def Y_i(reaction_rates=None, rate_arrays=None, temp_arrays=None):
    """
    Takes an array R(T) and returns final abundance Y for desired element. 
    Should use temperatures at same key network temperatures. 
    """

    #network = 'key_PArthENoPE_discrete'
    network = 'key_PRIMAT_2023'

    if reaction_rates is not None:
        abd_model = AbundanceModel(NuclearRates(nuclear_net=network, 
                                                reaction_rates=reaction_rates, 
                                                rate_arrays=rate_arrays, 
                                                temp_arrays=temp_arrays))
    else:
        abd_model = AbundanceModel(NuclearRates(nuclear_net=network))
        
    sol = abd_model(
        rho_g_vec, rho_nu_vec, 
        rho_NP_vec, p_NP_vec,
        t_vec=t_vec, a_vec=a_vec, 
        eta_fac = jnp.asarray(1.), 
        tau_n_fac = jnp.asarray(1.), 
        nuclear_rates_q = jnp.zeros(12),
        save_history=False 
    )
    return sol
    
reaction_rates = [1, 2, 3]          # Indices for dphe3g, ddhe3n, ddtp respectively. Change to include desired reactions
temperatures = [T_primat for i in range(len(reaction_rates))]
D = []
He3 = []
He4 = []

# Pass each rate sample through LINX and save abundances

def Yp(He4, H):
    m_he4 = 4.0026     # amu
    m_h = 1.0078       # amu
    return m_he4 * He4 / (m_he4 * He4 + m_h * H)

for i in range(num_samples):
    dphe3g = dphe3g_R_samples[i]
    ddhe3n = ddhe3n_R_samples[i]
    ddtp = ddtp_R_samples[i]
    
    sol = Y_i(reaction_rates=reaction_rates, rate_arrays=[dphe3g, ddhe3n, ddtp], temp_arrays=temperatures)
    D.append(sol[2]/sol[1])
    He3.append(sol[4]/sol[1])
    He4.append(Yp(sol[5], sol[1]))

# Output abundances as txt

with open(out_dir+'gp_abundances.txt', 'w') as f:
    f.write('D/H\tHe3/H\tYp')
    for i in range(num_samples):
        f.write('\n'+str(D[i])+'\t'+str(He3[i])+'\t'+str(He4[i]))
        
# Print statistics of abundance posteriors

print('Primordial abundance statistics for '+str(num_samples)+' samples:')

print('D/H mean: ', np.mean(D))
print('D/H sigma: ', np.std(D))
print('He3/H mean: ', np.mean(He3))
print('He3/H sigma: ', np.std(He3))
print('Yp mean: ', np.mean(He4))
print('Yp sigma: ', np.std(He4))


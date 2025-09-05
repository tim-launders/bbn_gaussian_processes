# This file computes primordial abundances from sampling from the GP posterior for ddHe3n S-factor. 


import sys
sys.path.append('.')

import numpy as np
import jax.numpy as jnp
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from sample_rates import dd_R
from gaussian_processes.kernels import Constant, SE, Matern
from gaussian_processes.gp import GPR
from LINX.linx.data.nuclear_rates.key_PArthENoPE import ddHe3n_frwrd_rate
from LINX.linx.abundances import AbundanceModel
from LINX.linx.background import BackgroundModel
from LINX.linx.nuclear import NuclearRates
from LINX.linx.thermo import T_g


# Import ddHe3n S-factor data


data_dir = './data/ddHe3n/'

sets = ['BR90', 'GR95', 'SC72', 'Leo06', 'KR87B', 'KR87M']
colors = ['blue', 'cyan', 'brown', 'red', 'indigo', 'seagreen']

all_E = []
all_S = []
all_total_error = []
all_systematic_error = []
points = []

for set in sets:
    energy, cross, S, total, sys = np.loadtxt(data_dir+set+'.txt', unpack=True)
    all_E.append(energy)
    all_S.append(S)
    all_total_error.append(total)
    all_systematic_error.append(sys)
    points.append(len(S))

E = np.concatenate(all_E)
logE = np.log(E)
S = np.concatenate(all_S)
total_error = np.concatenate(all_total_error)
systematic_error = np.concatenate(all_systematic_error)         # Multiply by 0 for uncorrelated case


# Import theory curve for GP mean


file = './data/ddhe3n_theory_curve.csv'
theory_E, theory_S = np.loadtxt(file, delimiter=',', unpack=True)
sort_idx = np.argsort(theory_E)
x_sorted = np.log(theory_E[sort_idx])
y_sorted = theory_S[sort_idx]
theory_mean = UnivariateSpline(x=x_sorted, y=y_sorted, s=0)
theory_mean = lambda x: 0                                       # Uncomment when using zero mean prior


# Load in parameters for the kernel and fit the GP


# Uncomment for desired kernel hyperparameters, previously fitted. LDO optimization used unless otherwise stated.

#hyper = [4.95230230E+04, 1.63550757E+01, 4.65902731E-03, 2.14710930E+03]         # Theory mean
hyper = [2.43637577E+05, 3.43956298E+01, 4.60471861E-03, 2.17241095E+03]         # Zero mean
#hyper = [3.25565503E+06, 7.05118447E+01, 4.17345785E-06, 2.39691819E+06]         # No correlations, zero mean
#hyper = [9.90368587E-01, 4.91453150E+00, 3.05801581E-03, 3.28278797E+03]         # MLE, zero mean
kernel = Constant(hyper[0]) * SE(hyper[1]) + Constant(hyper[2]) * Matern(hyper[3], nu=0.25)

gp = GPR(kernel, mean_func=theory_mean)
gp.fit(logE, S, total_error, systematic_error, points)


# Draw samples


num_samples = 10000
num_points = 1000
E_pred = jnp.logspace(jnp.log10(5e-3), jnp.log10(3.2), num_points)
S_mean = gp.predict(jnp.log(E_pred))
samples = gp.draw_samples(jnp.log(E_pred), num_samples)


# Define function to select nearest value in an array


def nearest_index(array, value):
    return jnp.abs(array - value).argmin()


# For each sample, compute the rate


Ti = 1e6
Tf = 1e10
num_T = 500
T_pred = jnp.logspace(jnp.log10(Ti), jnp.log10(Tf), num_T)


# Load in PRIMAT data to directly compare with GP posterior at the same temperatures


filepath = './LINX/linx/data/nuclear_rates/key_PRIMAT_2023/ddHe3n.txt'
T9, r_primat, f = np.loadtxt(filepath, unpack=True)
T_primat = T9 * 1e9
first = 17
last = 51
T_pred = T_primat                   # Comment to use previously defined temperature array
num_T = len(T_primat)

R_samples = []
R_samples_primat = []
for sample in samples: 
    rate = jnp.array([dd_R(t, E_pred, sample) for t in T_pred])
    R_samples.append(rate)
    #rate_primat = jnp.array([dd_R(t, E_pred, sample) for t in T_primat])
    R_samples_primat.append(rate)
R_samples = jnp.array(R_samples)
R_samples_primat = jnp.array(R_samples_primat)


# Calculate mean and 1 sigma at each temperature


gp_R_mean = jnp.mean(R_samples, axis=0)
gp_R_sigma = jnp.std(R_samples, axis=0)
gp_R_mean_primat = jnp.mean(R_samples_primat, axis=0)


# Plot rate posterior with 1 sigma envelope


Ti_plot = 2e7
Tf_plot = 2.5e9
Ti_plot_index = nearest_index(T_pred, Ti_plot)
Tf_plot_index = nearest_index(T_pred, Tf_plot)
T_plot = T_pred[Ti_plot_index:Tf_plot_index]
gp_R_mean_plot = gp_R_mean[Ti_plot_index:Tf_plot_index]
gp_R_sigma_plot = gp_R_sigma[Ti_plot_index:Tf_plot_index]

figure_dir = './abundance_calculations/figures/ddHe3n/'

gp_sigma_ratio = gp_R_sigma_plot / gp_R_mean_plot

#plt.plot(T_plot, 1 - gp_sigma_ratio, color='black', label=r'GP $1\sigma$ envelope')        # Uncomment for lines to outline envelope
#plt.plot(T_plot, 1 + gp_sigma_ratio, color='black')
plt.fill_between(T_plot, 1-gp_sigma_ratio, 1+gp_sigma_ratio, color='black', alpha=0.3, linewidth=0, label=r'GP $1\sigma$ envelope')
plt.xscale('log')
plt.xlabel('T (K)')
plt.ylabel('Rate / GP mean rate')
plt.title(f'GP Rate Posterior, {num_samples} samples')
plt.savefig(figure_dir+'rate_envelope.png')
plt.clf()

#plt.plot(T_plot, 1 - gp_sigma_ratio, color='black', label=r'GP $1\sigma$ envelope')
#plt.plot(T_plot, 1 + gp_sigma_ratio, color='black')
plt.fill_between(T_plot, 1-gp_sigma_ratio, 1+gp_sigma_ratio, color='black', alpha=0.3, linewidth=0, label=r'GP $1\sigma$ envelope')
for i in range(5):
    sample = R_samples[i][Ti_plot_index:Tf_plot_index]
    plt.plot(T_plot, sample/gp_R_mean_plot)
plt.xscale('log')
plt.xlabel('T (K)')
plt.ylabel('Rate / GP mean rate')
plt.title(f'GP Rate Posterior, {num_samples} samples')
plt.savefig(figure_dir+'rate_envelope_samples.png')
plt.clf()


# Add PRIMAT and PArthENoPE rates to plots


primat_up = np.multiply(r_primat, f)
primat_down = np.multiply(r_primat, 2-f)
primat_ratio_up = primat_up / gp_R_mean_primat
primat_ratio_down = primat_down / gp_R_mean_primat

parth_up = np.array([ddHe3n_frwrd_rate(t, 1) for t in T_plot])
parth_down = np.array([ddHe3n_frwrd_rate(t, -1) for t in T_plot])
parth_mean = np.array([ddHe3n_frwrd_rate(t, 0) for t in T_plot])
parth_ratio_up = parth_up / gp_R_mean_plot
parth_ratio_down = parth_down / gp_R_mean_plot

#plt.plot(T_plot, 1 - gp_sigma_ratio, color='black', label=r'GP $1\sigma$ envelope')
#plt.plot(T_plot, 1 + gp_sigma_ratio, color='black')
plt.fill_between(T_plot, 1-gp_sigma_ratio, 1+gp_sigma_ratio, color='black', alpha=0.3, linewidth=0, label=r'GP $1\sigma$ envelope')
plt.fill_between(T_primat[first:last], primat_ratio_down[first:last], primat_ratio_up[first:last], color='red', alpha=0.5, linewidth=0, label=r'PRIMAT $1\sigma$ envelope')
plt.fill_between(T_plot, parth_ratio_down, parth_ratio_up, color='blue', alpha=0.5, linewidth=0, label=r'PArthENoPE $1\sigma$ envelope')
plt.legend()
plt.xscale('log')
plt.xlabel('T (K)')
plt.ylabel('Rate / GP mean rate')
plt.title(f'GP Rate Posterior, {num_samples} samples')
plt.savefig(figure_dir+'rate_envelope_comp.png')
plt.clf()

#plt.plot(T_plot, 1 - gp_sigma_ratio, color='black', label=r'GP $1\sigma$ envelope')
#plt.plot(T_plot, 1 + gp_sigma_ratio, color='black')
plt.fill_between(T_plot, 1-gp_sigma_ratio, 1+gp_sigma_ratio, color='black', alpha=0.3, linewidth=0, label=r'GP $1\sigma$ envelope')
plt.fill_between(T_primat[first:last], primat_ratio_down[first:last], primat_ratio_up[first:last], color='red', alpha=0.5, linewidth=0, label=r'PRIMAT $1\sigma$ envelope')
plt.fill_between(T_plot, parth_ratio_down, parth_ratio_up, color='blue', alpha=0.5, linewidth=0, label=r'PArthENoPE $1\sigma$ envelope')
for i in range(5):
    sample = R_samples[i][Ti_plot_index:Tf_plot_index]
    plt.plot(T_plot, sample/gp_R_mean_plot)
plt.legend()
plt.xscale('log')
plt.xlabel('T (K)')
plt.ylabel('Rate / GP mean rate')
plt.title(f'GP Rate Posterior, {num_samples} samples')
plt.savefig(figure_dir+'rate_envelope_comp_samples.png')
plt.clf()


# Output mean and 1 sigma GP rates as a txt file


out_dir = './data/output/ddHe3n/'

with open(out_dir+'gp_rates.txt', 'w') as f:
    f.write('T (K)\tMean Rate\t1 sigma')
    for i in range(num_T):
        f.write('\n'+str(T_pred[i])+'\t'+str(gp_R_mean[i])+'\t'+str(gp_R_sigma[i]))
        
        
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
    
reaction_rates = [2]
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
    sample = R_samples_primat[i]
    sol = Y_i(reaction_rates=reaction_rates, rate_arrays=[sample], temp_arrays=temperatures)
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


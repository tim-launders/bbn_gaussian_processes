# This file computes primordial abundances from sampling from the GP posterior for dpHe3g S-factor. 


import sys
sys.path.append('.')

import numpy as np
import jax.numpy as jnp
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sample_rates import dp_R
from gaussian_processes.kernels import Constant, SE, Dot, Matern
from gaussian_processes.gp import GPR
from LINX.linx.data.nuclear_rates.key_PArthENoPE import dpHe3g_frwrd_rate
from LINX.linx.abundances import AbundanceModel
from LINX.linx.background import BackgroundModel
from LINX.linx.nuclear import NuclearRates
from LINX.linx.thermo import T_g


# Import ddHe3n S-factor data


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
    energy, cross, S, total, sys = np.loadtxt(data_dir+set+'.txt', unpack=True)
    all_E.append(energy)
    all_S.append(S)
    all_total_error.append(total)
    all_systematic_error.append(sys)
    all_statistical_error.append(np.sqrt(total**2 - sys**2))
    points.append(len(S))

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


# Load in parameters for the kernel and fit the GP


hyper = [2.31789296E+03, 7.60324923E+00, 2.66962082E-02, 3.74584365E+02]
kernel = Constant(hyper[0]) * SE(hyper[1]) + Constant(hyper[2]) * Matern(hyper[3], nu=0.25)

gp = GPR(kernel, noise=0)
gp.fit(logE, logS, log_total_error, log_systematic_error, points)
cov_mat = gp.calculate_covariance(log_total_error, log_systematic_error, points)


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

gp.set_covariance(cov_mat)


# Draw samples


num_samples = 10000
num_points = 1000
E_pred = jnp.logspace(np.log10(2.5e-3), np.log10(2), num_points)
logS_mean = gp.predict(jnp.log(E_pred))
S_mean = jnp.exp(logS_mean)
log_samples = gp.draw_samples(jnp.log(E_pred), num_samples)
samples = jnp.exp(log_samples)


# Define function to select nearest value in an array


def nearest_index(array, value):
    return jnp.abs(array - value).argmin()


# For each sample, compute the rate
Ti = 1e6
Tf = 1e10
num_T = 500
T_pred = jnp.logspace(jnp.log10(Ti), jnp.log10(Tf), num_T)


# Load in PRIMAT data to directly compare with GP posterior at correct temperatures


filepath = './LINX/linx/data/nuclear_rates/key_PRIMAT_2023/dpHe3g.txt'
T9, r_primat, f = np.loadtxt(filepath, unpack=True)
T_primat = T9 * 1e9
first = 17
last = 51
T_pred = T_primat                   # Comment to use previously defined temperature array
num_T = len(T_primat)

R_samples = []
R_samples_primat = []
for sample in samples: 
    rate = jnp.array([dp_R(t, E_pred, sample) for t in T_pred])
    R_samples.append(rate)
    #rate_primat = jnp.array([dp_R(t, E_pred, sample) for t in T_primat])
    R_samples_primat.append(rate)
R_samples = jnp.array(R_samples)
R_samples_primat = jnp.array(R_samples_primat)


# Calculate mean and 1 sigma at each temperature


gp_R_mean = jnp.mean(R_samples, axis=0)
gp_R_sigma = jnp.std(R_samples, axis=0)
gp_R_mean_primat = jnp.mean(R_samples_primat, axis=0)


# Plot histogram of samples at a given temperature (check for non-Gaussianity)


figure_dir = './abundance_calculations/figures/dpHe3g/'

def get_fit(data, x_min, x_max, num_bins=50):
    """
    Returns the binned dataset along with the gaussian fit. 
    """
    x, bins, bin_width = bin_sample(data, x_min, x_max, num_bins)

    def gaussian(x, b, c):
        """
        Returns gaussian parametrized by b, c at x.
        """
        return bin_width * num_samples / np.sqrt(2*np.pi*(c**2)) * np.exp(-(x-b)**2/(2*c**2))

    p0 = (np.mean(data), np.std(data))
    fit, cov = curve_fit(gaussian, x, bins, p0)

    return x, bins, fit, cov

def bin_sample(data, x_min, x_max, num_bins=50):
    """
    Bins a dataset into 50 bins. 
    """
    bins, edges, patches = plt.hist(data, num_bins, (x_min, x_max))
    plt.clf()
    x = np.zeros(num_bins)
    dx = edges[1] - edges[0]
    for i in range(num_bins):
        x[i] = edges[i] + dx/2

    return x, bins, dx

def gaussian_fit(x, b, c, bin_width):
    return bin_width * num_samples / np.sqrt(2*np.pi*c**2) * np.exp(-(x-b)**2/(2*c**2))

plot_ind = 26
T_hist = T_primat[plot_ind]
R_plot = R_samples[:,plot_ind]

gp_mean_R = dp_R(T_hist, E_pred, S_mean)

rx, rg, r_fit, r_cov = get_fit(R_plot, min(R_plot), max(R_plot))
rx_min = rx[0] - 1e-3
rx_max = rx[-1] + 1e-3
xr = np.linspace(rx_min, rx_max, 200)

bins, edges, patches = plt.hist(R_plot, bins=50, range=(min(R_plot), max(R_plot)), histtype='step')
dx = edges[1]-edges[0]
plt.plot(xr, gaussian_fit(xr, *r_fit, dx))
plt.axvline(x=gp_mean_R, color='black', linewidth=1, linestyle='dashed')
plt.xlabel(r'Rate (cm$^{-3}$mol$^{-1}$s$^{-1}$)')
plt.ylabel('Number')
plt.title('T=0.1 GK')
#plt.savefig(figure_dir+'dprate_dist.png')
plt.clf()


# Plot rate posterior with 1 sigma envelope


Ti_plot = 2e7
Tf_plot = 2.5e9
Ti_plot_index = nearest_index(T_pred, Ti_plot)
Tf_plot_index = nearest_index(T_pred, Tf_plot)
T_plot = T_pred[Ti_plot_index:Tf_plot_index]
gp_R_mean_plot = gp_R_mean[Ti_plot_index:Tf_plot_index]
gp_R_sigma_plot = gp_R_sigma[Ti_plot_index:Tf_plot_index]

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


# Add PRIMAT and PArthENoPE rates


primat_up = np.multiply(r_primat, f)
primat_down = np.multiply(r_primat, 2-f)
primat_ratio_up = primat_up / gp_R_mean_primat
primat_ratio_down = primat_down / gp_R_mean_primat

parth_up = np.array([dpHe3g_frwrd_rate(t, 1) for t in T_plot])
parth_down = np.array([dpHe3g_frwrd_rate(t, -1) for t in T_plot])
parth_mean = np.array([dpHe3g_frwrd_rate(t, 0) for t in T_plot])
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
    
reaction_rates = [1]
D = []
He3 = []
He4 = []


# Pass each rate sample through LINX and save abundances


def Yp(He4, H):
    m_he4 = 4.0026     # amu
    m_h = 1.0078       # amu
    return m_he4 * He4 / (m_he4 * He4 + m_h * H)

temperatures = [T_primat]

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


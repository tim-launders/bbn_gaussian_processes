# Takes abundance posterior and plots histograms with a Gaussian fit. 


import sys
sys.path.append('.')

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from LINX.linx.abundances import AbundanceModel
from LINX.linx.background import BackgroundModel
from LINX.linx.nuclear import NuclearRates
from LINX.linx.thermo import T_g
import jax.numpy as jnp

# Change to what is used
print('\nVarying reaction: ddtp')
print('\nKernel choice: SE + Matern-1/4')
reaction_id = 3


# Load in data and define functions for the fit


data_file = './data/ddtp/gp_abundances.txt'
fig_out = './visualization/figures/'

d, he3, he4 = np.loadtxt(data_file, unpack=True, skiprows=1)

num_points = len(d)

def get_fit(data, x_min, x_max, num_bins=50):
    """
    Returns the binned dataset along with the gaussian fit. 
    """
    x, bins, bin_width = bin_sample(data, x_min, x_max, num_bins)

    def gaussian(x, b, c):
        """
        Returns gaussian parametrized by b, c at x.
        """
        return bin_width * num_points / np.sqrt(2*np.pi*(c**2)) * np.exp(-(x-b)**2/(2*c**2))

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


# Perform the Gaussian fits


dx, dg, d_fit, d_cov = get_fit(d, min(d), max(d))
he3x, he3g, he3_fit, he3_cov = get_fit(he3, min(he3), max(he3))
he4x, he4g, he4_fit, he4_cov = get_fit(he4, min(he4), max(he4))

xd_min = dx[0] - 1e-8
xd_max = dx[-1] + 1e-8
xhe3_min = he3x[0] - 1e-8
xhe3_max = he3x[-1] + 1e-8
xhe4_min = he4x[0] - 1e-8
xhe4_max = he4x[-1] + 1e-8

xd = np.linspace(xd_min, xd_max, 200)
xhe3 = np.linspace(xhe3_min, xhe3_max, 200)
xhe4 = np.linspace(xhe4_min, xhe4_max, 200)


# Print results of the fit 


print('Gaussian process sampling:')
print('D/H mean: ', np.mean(d))
print('D/H fit mean: ', d_fit[0])
print('D/H sigma: ', np.std(d))
print('D/H fit sigma: ', d_fit[1])
print('He3/H mean: ', np.mean(he3))
print('He3/H fit mean: ', he3_fit[0])
print('He3/H sigma: ', np.std(he3))
print('He3/H fit sigma: ', he3_fit[1])
print('Yp mean: ', np.mean(he4))
print('Yp fit mean: ', he4_fit[0])
print('Yp sigma: ', np.std(he4))
print('Yp fit sigma: ', he4_fit[1])


# Calculate base key_PRIMAT abundances for comparison

rates_q = jnp.zeros(12)
rates_q_up = np.zeros(12)
rates_q_down = np.zeros(12)
rates_q_up[reaction_id] = 1
rates_q_down[reaction_id] = -1
rates_q_up = jnp.array(rates_q_up)
rates_q_down = jnp.array(rates_q_down)

bkg_model = BackgroundModel()
t_vec, a_vec, rho_g_vec, rho_nu_vec, rho_NP_vec, p_NP_vec, Neff_vec = bkg_model(jnp.asarray(0.))
T_g_vec = T_g(rho_g_vec)

abd_model = AbundanceModel(NuclearRates(nuclear_net='key_PRIMAT_2023'))

sol = abd_model(rho_g_vec, rho_nu_vec, 
                rho_NP_vec, p_NP_vec, 
                t_vec=t_vec, a_vec=a_vec, 
                eta_fac = jnp.asarray(1.), 
                tau_n_fac = jnp.asarray(1.), 
                nuclear_rates_q = rates_q,
                save_history=False)

sol_up = abd_model(rho_g_vec, rho_nu_vec, 
                rho_NP_vec, p_NP_vec, 
                t_vec=t_vec, a_vec=a_vec, 
                eta_fac = jnp.asarray(1.), 
                tau_n_fac = jnp.asarray(1.), 
                nuclear_rates_q = rates_q_up,
                save_history=False)
                
sol_down = abd_model(rho_g_vec, rho_nu_vec, 
                rho_NP_vec, p_NP_vec, 
                t_vec=t_vec, a_vec=a_vec, 
                eta_fac = jnp.asarray(1.), 
                tau_n_fac = jnp.asarray(1.), 
                nuclear_rates_q = rates_q_down,
                save_history=False)

def Yp(He4, H):
    m_he4 = 4.0026     # amu
    m_h = 1.0078       # amu
    return m_he4 * He4 / (m_he4 * He4 + m_h * H)

primat_dh = sol[2]/sol[1]
primat_he3h = sol[4]/sol[1]
primat_yp = Yp(sol[5], sol[1])

primat_dh_up = sol_up[2]/sol_up[1]
primat_he3h_up = sol_up[4]/sol_up[1]
primat_yp_up = Yp(sol_up[5], sol_up[1])

primat_dh_down = sol_down[2]/sol_down[1]
primat_he3h_down = sol_down[4]/sol_down[1]
primat_yp_down = Yp(sol_down[5], sol_down[1])


print('PRIMAT D/H mean: ', primat_dh)
print('PRIMAT D/H 1 sigma up: ', primat_dh_up)
print('PRIMAT D/H 1 sigma down: ', primat_dh_down)
print('PRIMAT He3/H mean: ', primat_he3h)
print('PRIMAT He3/H 1 sigma up: ', primat_he3h_up)
print('PRIMAT He3/H 1 sigma down: ', primat_he3h_down)
print('PRIMAT Yp mean: ', primat_yp)
print('PRIMAT Yp 1 sigma up: ', primat_yp_up)
print('PRIMAT Yp 1 sigma down: ', primat_yp_down)



# Plots


def gaussian_fit(x, b, c, bin_width):
    return bin_width * num_points / np.sqrt(2*np.pi*c**2) * np.exp(-(x-b)**2/(2*c**2))

bins, edges, patches = plt.hist(d, bins=50, range=(min(d), max(d)), histtype='step', label='LINX output')
dx = edges[1]-edges[0]
plt.plot(xd, gaussian_fit(xd, *d_fit, dx), label='Gaussian fit')
plt.axvline(x=primat_dh, color='g', linestyle='dashed', linewidth=1, label='key_PRIMAT_2023')
plt.axvspan(primat_dh_down, primat_dh_up, color='g', alpha=0.2, linewidth=0)
plt.legend()
plt.xlabel('D/H')
plt.ylabel('Number')
plt.title('D/H Distribution, '+str(num_points)+' samples')
plt.savefig(fig_out+'d_h_gaussian.png')
plt.clf()

bins, edges, patches = plt.hist(he3, bins=50, range=(min(he3), max(he3)), histtype='step', label='LINX output')
dx = edges[1]-edges[0]
plt.plot(xhe3, gaussian_fit(xhe3, *he3_fit, dx), label='Gaussian fit')
plt.axvline(x=primat_he3h, color='g', linestyle='dashed', linewidth=1, label='key_PRIMAT_2023')
plt.axvspan(primat_he3h_down, primat_he3h_up, color='g', alpha=0.2, linewidth=0)
#plt.legend()
plt.xlabel(r'$^3$He/H')
plt.ylabel('Number')
plt.title(r'$^3$He/H Distribution, '+str(num_points)+' samples')
plt.savefig(fig_out+'he3_h_gaussian.png')
plt.clf()

bins, edges, patches = plt.hist(he4, bins=50, range=(min(he4), max(he4)), histtype='step', label='LINX output')
dx = edges[1]-edges[0]
plt.plot(xhe4, gaussian_fit(xhe4, *he4_fit, dx), label='Gaussian fit')
plt.axvline(x=primat_yp, color='g', linestyle='dashed', linewidth=1, label='key_PRIMAT_2023')
plt.axvspan(primat_yp_down, primat_yp_up, color='g', alpha=0.2, linewidth=0)
#plt.legend()
plt.xlabel(r'Y$_\text{P}$')
plt.ylabel('Number')
plt.title(r'$\text{Y_p}$ Distribution, '+str(num_points)+' samples')
plt.savefig(fig_out+'yp_gaussian.png')
plt.clf()


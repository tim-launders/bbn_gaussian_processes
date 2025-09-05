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
print('\nVarying reactions: ddhe3n, ddtp')
print('\nKernel choice: SE + Matern-1/4')
print('\nOther rates used: PRIMAT')


# Load in data and define functions for the fit


data_file = './data/output/deuterium/gp_abundances.txt'
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


# Input base PRIMAT and PArthENoPE abundances for comparison



primat_dh = 2.43473e-5
primat_he3h = 1.02777e-5
primat_yp = 0.245830

primat_dh_sigma = 0.01892e-5
primat_he3h_sigma = 0.00366e-5
primat_yp_sigma = 0.0000217

primat_dh_up = primat_dh + primat_dh_sigma
primat_he3h_up = primat_he3h + primat_he3h_sigma
primat_yp_up = primat_yp + primat_yp_sigma

primat_dh_down = primat_dh - primat_dh_sigma
primat_he3h_down = primat_he3h - primat_he3h_sigma
primat_yp_down = primat_yp - primat_yp_sigma

parth_dh = 2.50279e-5
parth_he3h = 1.02537e-5
parth_yp = 0.245722

parth_dh_sigma = 0.01781e-5
parth_he3h_sigma = 0.00330e-5
parth_yp_sigma = 0.0000185

parth_dh_up = parth_dh + parth_dh_sigma
parth_he3h_up = parth_he3h + parth_he3h_sigma
parth_yp_up = parth_yp + parth_yp_sigma

parth_dh_down = parth_dh - parth_dh_sigma
parth_he3h_down = parth_he3h - parth_he3h_sigma
parth_yp_down = parth_yp - parth_yp_sigma



# Plots



def gaussian_fit(x, b, c, bin_width):
    return bin_width * num_points / np.sqrt(2*np.pi*c**2) * np.exp(-(x-b)**2/(2*c**2))

bins, edges, patches = plt.hist(d, bins=50, range=(min(d), max(d)), histtype='step', label='LINX output')
dx = edges[1]-edges[0]
plt.plot(xd, gaussian_fit(xd, *d_fit, dx), label='Gaussian fit')
plt.axvline(x=primat_dh, color='g', linestyle='dashed', linewidth=1, label='key_PRIMAT_2023')
plt.axvspan(primat_dh_down, primat_dh_up, color='g', alpha=0.2, linewidth=0)
plt.axvline(x=parth_dh, color='m', linestyle='dashed', linewidth=1, label='key_PArthENoPE')
plt.axvspan(parth_dh_down, parth_dh_up, color='m', alpha=0.2, linewidth=0)
plt.legend()
plt.xlabel('D/H')
plt.ylabel('Number')
plt.title('D/H Distribution, '+str(num_points)+' samples')
plt.savefig(fig_out+'dd_d_h_gaussian.png')
plt.clf()

bins, edges, patches = plt.hist(he3, bins=50, range=(min(he3), max(he3)), histtype='step', label='LINX output')
dx = edges[1]-edges[0]
plt.plot(xhe3, gaussian_fit(xhe3, *he3_fit, dx), label='Gaussian fit')
plt.axvline(x=primat_he3h, color='g', linestyle='dashed', linewidth=1, label='key_PRIMAT_2023')
plt.axvspan(primat_he3h_down, primat_he3h_up, color='g', alpha=0.2, linewidth=0)
plt.axvline(x=parth_he3h, color='m', linestyle='dashed', linewidth=1, label='key_PArthENoPE')
plt.axvspan(parth_he3h_down, parth_he3h_up, color='m', alpha=0.2, linewidth=0)
#plt.legend()
plt.xlabel(r'$^3$He/H')
plt.ylabel('Number')
plt.title(r'$^3$He/H Distribution, '+str(num_points)+' samples')
plt.savefig(fig_out+'dd_he3_h_gaussian.png')
plt.clf()

bins, edges, patches = plt.hist(he4, bins=50, range=(min(he4), max(he4)), histtype='step', label='LINX output')
dx = edges[1]-edges[0]
plt.plot(xhe4, gaussian_fit(xhe4, *he4_fit, dx), label='Gaussian fit')
plt.axvline(x=primat_yp, color='g', linestyle='dashed', linewidth=1, label='key_PRIMAT_2023')
plt.axvspan(primat_yp_down, primat_yp_up, color='g', alpha=0.2, linewidth=0)
plt.axvline(x=parth_yp, color='m', linestyle='dashed', linewidth=1, label='key_PArthENoPE')
plt.axvspan(parth_yp_down, parth_yp_up, color='m', alpha=0.2, linewidth=0)
#plt.legend()
plt.xlabel(r'Y$_\text{P}$')
plt.ylabel('Number')
plt.title(r'Y$_\text{P}$ Distribution, '+str(num_points)+' samples')
plt.savefig(fig_out+'dd_yp_gaussian.png')
plt.clf()


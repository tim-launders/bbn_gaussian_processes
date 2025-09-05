# Compares correlated and uncorrelated GPR on S-factor data


import sys
sys.path.append('.')

import jax
import numpy as np
import jax.numpy as jnp
import scipy
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from gaussian_processes.kernels import Constant, SE, Dot, Matern, RationalQuadratic
from gaussian_processes.optimizers import MarginalLikelihoodOptimizer, LOOOptimizer, kfoldCVOptimizer, LeaveDatasetOutOptimizer
from gaussian_processes.gp import GPR
import matplotlib.pyplot as plt
from scipy.special import kv

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

hyper_uncorrelated = [3.25565503E+06, 7.05118447E+01, 4.17345785E-06, 2.39691819E+06]
hyper_correlated = [2.43637577E+05, 3.43956298E+01, 4.60471861E-03, 2.17241095E+03]         

kernel_uncorrelated = Constant(hyper_uncorrelated[0]) * SE(hyper_uncorrelated[1]) + Constant(hyper_uncorrelated[2]) * Matern(hyper_uncorrelated[3], nu=0.25)
kernel_correlated = Constant(hyper_correlated[0]) * SE(hyper_correlated[1]) + Constant(hyper_correlated[2]) * Matern(hyper_correlated[3], nu=0.25)

gp_uncorrelated = GPR(kernel_uncorrelated)
gp_uncorrelated.fit(logE, S, total_error, 0*systematic_error, points)

gp_correlated = GPR(kernel_correlated)
gp_correlated.fit(logE, S, total_error, systematic_error, points)

E_pred = jnp.array([0.98,1.255,1.51,1.75,2.0,2.25,2.5,2.75,2.9,3.1])

mean_uncorrelated, sigma_uncorrelated = gp_uncorrelated.predict(jnp.log(E_pred), True)
mean_correlated, sigma_correlated = gp_correlated.predict(jnp.log(E_pred), True)

num_samples = 30
samples_uncorrelated = gp_uncorrelated.draw_samples(jnp.log(E_pred), num_samples)
samples_correlated = gp_correlated.draw_samples(jnp.log(E_pred), num_samples)

comp_S = jnp.array([0.2787,0.3185,0.3516,0.3759,0.4028,0.4301,0.4558,0.4749,0.4868,0.5089])
comp_sigma = jnp.array([0.009335,0.01067,0.01178,0.01259,0.01349,0.01441,0.01527,0.01591,0.01631,0.01705])

difference_uncorrelated = [(samples_uncorrelated[i] - comp_S)/comp_sigma for i in range(num_samples)]
difference_correlated = [(samples_correlated[i] - comp_S)/comp_sigma for i in range(num_samples)]

width = 1.5
fig = plt.figure(figsize=(12,8))
plt.plot(E_pred, difference_uncorrelated[0], color='black', label='Uncorrelated Samples', linewidth=width)
plt.plot(E_pred, difference_correlated[0], color='blue', label='Correlated Samples', linewidth=width)

for i in range(1, num_samples):
    plt.plot(E_pred, difference_correlated[i], color='blue', linewidth=width)
for i in range(1, num_samples):
    plt.plot(E_pred, difference_uncorrelated[i], color='black', linewidth=width)

plt.legend(fontsize=14, framealpha=1.0)

plt.xlabel('Energy (MeV)', fontsize=14)
plt.ylabel('Sample deviations', fontsize=14)
plt.title('Gaussian process sample deviations from SC72 data', fontsize=16)
plt.show()
plt.clf()

difference_correlated = np.array(difference_correlated)
print(np.mean(difference_correlated, axis=0))

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

corr_plot = difference_correlated[:,0]
corr_x, corr_g, corr_fit, corr_cov = get_fit(corr_plot, min(corr_plot), max(corr_plot))
corr_x_min = corr_x[0]-1e-3
corr_x_max = corr_x[-1]+1e-3
xcorr = np.linspace(corr_x_min, corr_x_max, 200)

bins, edges, patches = plt.hist(corr_plot, bins=50, range=(min(corr_plot), max(corr_plot)), histtype='step')
dx = edges[1] - edges[0]
plt.plot(xcorr, gaussian_fit(xcorr, *corr_fit, dx))
plt.xlabel('Sample S-factor deviations')
plt.ylabel('Number')
#plt.show()
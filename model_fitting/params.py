# Stores the parameters used by PArthENoPE to fit S(E) ddHe3n data

import numpy as np

# Parameters for degree 4 polynomial fit. Parameter units such that S(E) units in Mev b

a_mean = 0.05225
b_mean = 0.3655
c_mean = -0.1799
d_mean = 0.05832
f_mean = -0.007393

ddhe3n_mean = [a_mean, b_mean, c_mean, d_mean, f_mean]

ddtp_mean = [0.05520, 0.2151, -0.02555, 0, 0]

dphe3g_mean = [0.2121, 5.975, 5.463, -1.665, 0]

covar = 1e-6 * np.array([
    [0.0699, -0.600, 0.893, -0.454, 0.0738],
    [-0.600, 16.0, -26.2, 13.9, -2.30],
    [0.893, -26.2, 54.0, -30.6, 5.29],
    [-0.454, 13.9, -30.6, 18.1, -3.22],
    [0.0738, -2.30, 5.29, -3.22, 0.586]
])

def S(E, parameters):
    
    '''
    Calculates S(E) for given E using PArthENoPE parametrization.
    '''

    a = parameters[0]
    b = parameters[1]
    c = parameters[2]
    d = parameters[3]
    f = parameters[4]

    return a + b*E + c*E**2 + d*E**3 + f*E**2

def parth_S_ddhe3n(E):
    """
    Calculates the PArthENoPE best-fit S(E) for ddHe3n.
    """
    return S(E, ddhe3n_mean)

def parth_S_ddtp(E):
    """
    Calculates the PArthENoPE best-fit S(E) for ddtp.
    """
    return S(E, ddtp_mean)

def parth_S_dpHe3g(E):
    """
    Calculates the PArthENoPE best-fit S(E) for dpHe3g.
    """
    return S(E, dphe3g_mean)*1e-6

def delta_S(E):

    '''
    Calculates S(E) 1 sigma band for given E using PArthENoPE parametrization. 
    '''

    sum = 0
    for i in range(len(ddhe3n_mean)):
        for j in range(len(ddhe3n_mean)):
            sum += E**i * E**j * covar[i,j]
    return np.sqrt(sum)

# Constants for calculations

# Atomic unit, MeV
ma = 931.494061

# Deuterium nuclear mass, amu converted to MeV, PDG 2016
m_d = 2.01410 * ma

# Two-particle deuterium system reduced mass, MeV
m_d_r = m_d / 2

# Conversion constants
hbar = 6.582119 * 1e-16 * 1e-6 # s MeV
kB   = 1. / (1.160451812 * 1e4 * 1e6)  # MeV / K
c = 2.99792458e10 # cm / s
Na = 6.02214076 * 1e23 # mol^-1
barns_to_cm2 = 1e-24

# Fine-structure constant
aFS = 1./137.035999084 

# Gamow Energy for two-particle deuterium system
E_G = 2 * np.pi**2 * m_d_r * aFS**2

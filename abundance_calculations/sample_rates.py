# This file contains all functionality to go from S-factor to a thermally-averaged cross section (rate). 

import jax.numpy as jnp
import jax


# Constants for calculations


# Proton mass, MeV/c^2, PDG 2016
m_p = 938.2720813

# Deuteron mass, MeV/c^2, PDG 2016
m_d = 1875.612928

hbar = 6.582119 * 1e-16 * 1e-6 # s MeV
kB   = 1. / (1.160451812 * 1e4 * 1e6)  # MeV / K
c = 2.99792458e10 # cm / s
Na = 6.02214076 * 1e23 # mol^-1
barns_to_cm2 = 1e-24

# Fine-structure constant
aFS = 1./137.035999084 


def reduced_mass(m0, m1):
    """
    Computes reduced mass for desired reaction.
    
    Parameters
    ----------
    m0 : float
        Mass of first incident nuclei
    m1 : float
        Mass of second incident nuclei
    
    Returns
    -------
    mu: float
        Reduced mass of the system.
    """
    mu = (m0 * m1) / (m0 + m1)
    return mu
    

def E_gamow(mu, Z0, Z1):
    """
    Computes Gamow energy for desired reaction. 
    
    Parameters
    ----------
    mu : float
        Reduced mass of nuclei system. 
    Z0 : int or float
        Charge of first incident nuclei
    Z1 : int or float
        Charge of second incident nuclei
        
    Returns
    -------
    E_G : float
        Gamow energy for the reaction. 
    """
    E_G = 2 * jnp.pi**2 * mu * Z0 * Z1 * aFS**2
    return E_G


def A(T, mu):
    """
    Computes energy independent part of Boltzmann-Gamow kernel at temperature T.

    Parameters
    ----------
    T : int or float
        Temperature (K).
    mu : float
        Reduced mass of nuclei system. 
    
    Returns
    -------
    E_independent : float
        Energy independent part of the Boltzmann-Gamow kernel.
    """
    # Factor of c arises from mass in MeV/c^2
    E_independent = Na * jnp.sqrt(8 / (jnp.pi * mu)) * c * (kB * T)**(-3/2) * barns_to_cm2
    return E_independent


def K(E, T, E_G):
    """
    Computes energy dependent part of Boltzmann-Gamow kernel at energy E, temperature T. 
    This is the term that enters into the integral with the S-factor.

    Parameters
    ----------
    E : int or float
        Energy (MeV).
    T : int or float
        Temperature (K).
    E_G : float
        Gamow energy of nuclei system. 
    
    Returns
    -------
    E_dependent : float
        Energy dependent part of the Boltzmann-Gamow kernel.
    """
    E_dependent = jnp.exp(-E / (kB * T) - jnp.sqrt(E_G / E))
    return E_dependent


def R(T, energies, S, mu, E_G):
    """
    Performs integral numerically to compute thermally-averaged cross section. 

    Parameters
    ----------
    T : int or float
        Temperature (K) at which to compute the thermally-averaged cross section.
    energies : array-like
        Discretized energy values (MeV) to integrate over.
    S : array-like
        S-factor values (MeV barn) at the discretized energy values. 
    mu : float
        Reduced mass of nuclei system. 
    E_G : float
        Gamow energy of nuclei system. 
    
    Returns
    -------
    rate : float
        Thermally-averaged cross section (cm^3/mol/s).
    """
    energies = jnp.array(energies)
    S = jnp.array(S)
    integral = jax.scipy.integrate.trapezoid(K(energies, T, E_G) * S, energies)
    rate = A(T, mu) * integral
    return rate


def dd_R(T, energies, S):
    """
    Calculates reaction rate for deuterium-deuterium reactions.
    
    Parameters
    ----------
    T : int or float
        Temperature (K) at which to compute the thermally-averaged cross section.
    energies : array-like
        Discretized energy values (MeV) to integrate over.
    S : array-like
        S-factor values (MeV barn) at the discretized energy values. 
    
    Returns
    -------
    rate : float
        Thermally-averaged cross section (cm^3/mol/s).
    """
    mu = reduced_mass(m_d, m_d)
    E_G = E_gamow(mu, 1, 1)
    rate = R(T, energies, S, mu, E_G)
    return rate
    

def dp_R(T, energies, S):
    """
    Calculates reaction rate for deuterium-proton reactions.
    
    Parameters
    ----------
    T : int or float
        Temperature (K) at which to compute the thermally-averaged cross section.
    energies : array-like
        Discretized energy values (MeV) to integrate over.
    S : array-like
        S-factor values (MeV barn) at the discretized energy values. 
    
    Returns
    -------
    rate : float
        Thermally-averaged cross section (cm^3/mol/s).
    """
    mu = reduced_mass(m_d, m_p)
    E_G = E_gamow(mu, 1, 1)
    rate = R(T, energies, S, mu, E_G)
    return rate
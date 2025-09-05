# Takes raw experimental data and formats it for BBN use

import numpy as np

sets = ['CA02', 'GR62', 'GR63', 'MO20', 'MA97', 'SC97', 'WA63', 'TI19', 'TU21 3', 'TU21 4']
is_cross = [False, True, True, False, True, False, False, False, False, False]
is_total = [False, True, True, False, False, False, False, False, False, False]
is_percent = [True, False, False, False, True, True, True, True, True, True]
has_total_stat = [False, False, False, False, False, False, True, False, False, False]           # If total and statistical errors are provided

raw_dir = 'data/dpHe3g_raw/'
output_dir = 'data/dpHe3g/'

# Constants for calculations

# Proton mass, MeV/c^2, PDG 2016
m_p = 938.2720813

# Deuteron mass, MeV/c^2, PDG 2016
m_d = 1875.612928

# Fine-structure constant, PDG 2016
aFS = 1./137.035999139

# Reaction parameters

Z0 = 1.0
Z1 = 1.0
m0 = m_p
m1 = m_d

def S_to_cross(S, E):
    """
    Converts S-factor in MeV b to cross section in b.

    Parameters
    ----------
    S : float
        S-factor 
    E : float
        Energy in MeV.
    """
    mu = m0 * m1 / (m0 + m1)            # Reduced mass
    EG = 2 * np.pi**2 * mu * aFS**2.    # Gamow energy
    return S * np.exp(-np.sqrt(EG/E)) / E

def cross_to_S(cross, E):
    """
    Converts cross section in b to S-factor in MeV b.

    Parameters
    ----------
    cross : float
        Cross section in b
    E : float
        Energy in MeV. 
    """
    mu = m0 * m1 / (m0 + m1)            # Reduced mass
    EG = 2 * np.pi**2 * mu * aFS**2.    # Gamow energy
    return cross * np.exp(np.sqrt(EG/E)) * E

for i in range(len(sets)):
    set = sets[i]
    raw_file = raw_dir + set + '.txt'
    output_file = output_dir + set + '.txt'

    if is_total[i]:
        E, raw_value, raw_error = np.loadtxt(raw_file, skiprows=1, unpack=True)
        if is_percent[i]:
            percent_error = raw_error
        else:
            percent_error = raw_error / raw_value
        if is_cross[i]:
            cross = raw_value
            S = cross_to_S(cross, E)
        else:
            S = raw_value
            cross = S_to_cross(S, E)
        total_error = percent_error * S
        percent_sys = min(percent_error)
        systematic_error = percent_sys * S

    else:
        if has_total_stat[i]:
            E, raw_value, raw_total, raw_stat = np.loadtxt(raw_file, skiprows=1, unpack=True)
            raw_sys = np.sqrt(raw_total**2 - raw_stat**2)
        else:
            E, raw_value, raw_stat, raw_sys = np.loadtxt(raw_file, skiprows=1, unpack=True)
        if is_percent[i]:
            percent_sys = raw_sys
            if has_total_stat[i]:
                raw_stat = raw_stat * raw_value     # Convert statistical error from percentage  
        else:
            percent_sys = raw_sys / raw_value
        if is_cross[i]:
            cross = raw_value
            S = cross_to_S(cross, E)
        else:
            S = raw_value
            cross = S_to_cross(S, E)
        systematic_error = percent_sys * S
        total_error = np.sqrt(raw_stat**2 + systematic_error**2)        # Statistical error not reported as percentage

    with open(output_file, 'w') as f:
        f.write('#E_cm (MeV)\tCross Section (b)\tS (MeV b)\tTotal Error (MeV b)\tSystematic Error (MeV b)')
        for j in range(len(E)):
            f.write(f'\n{E[j]:.6f}\t{cross[j]:.6e}\t{S[j]:.6e}\t{total_error[j]:.6e}\t{systematic_error[j]:.6e}')

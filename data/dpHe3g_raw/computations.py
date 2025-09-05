import numpy as np
import jax.numpy as jnp

m_d = 1876  # MeV
m_p = 938   # MeV

def dp_cm(p_cm):

    # Computes center of mass energy when protons are fired at deuterium. 

    return m_d / (m_p + m_d) * p_cm

energies = np.array([0.0241, 0.0244, 0.027, 0.0302, 0.0334, 0.0345, 0.0362, 0.393, 
                     0.041, 0.0422, 0.0474, 0.0481])

energies = np.array([0.275, 0.580, 0.755, 0.985, 1.750])

energies = np.array([0.01494, 0.02499, 0.03498, 0.04489, 0.05494, 0.06493, 0.07491])

print(dp_cm(energies))
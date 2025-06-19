# This file contains support functions for use in the Gaussian processes module. 

import jax
import jax.numpy as jnp
from scipy.special import kv
from jax import custom_jvp, pure_callback

# Function to check if an input value is a positive half-integer for use in the Matern kernel. 

def is_positive_half_integer(x, tolerance=1e-8):
    """
    Checks if input value is a positive half-integer value.

    Parameters
    ----------
    x : int or float
        Value to check. 
    tolerance : float, default=1e-8
        Tolerance in evaluation to account for floating-point precision. 
    """
    is_half_integer = abs((x * 2) % 2 - 1) < tolerance
    is_positive = x > tolerance
    return jnp.logical_and(is_half_integer, is_positive)

# Functions to adapt the modified Bessel function of the second kind of arbitrary order to JAX. 

@custom_jvp
def mod_bessel(x, nu):
    """
    Adapts the modified Bessel function of the second kind from scipy to JAX.

    Parameters
    ----------
    nu : float
        The order of the modified Bessel function. 
    x : array_like of shape (n_points_1, n_points_2)
        The input values for which to compute the modified Bessel function. 
    """
    result_shape_dtype = jax.ShapeDtypeStruct(shape=x.shape, dtype=jnp.float64)
    return pure_callback(
        lambda x: kv(nu, x),
        result_shape_dtype,
        x,
        vmap_method='broadcast_all',
    )

@mod_bessel.defjvp
def mod_bessel_jvp(primals, tangents):
    """
    Defines the differentiation rule for the modified Bessel function of the second kind.
    This uses the recurrence relation for the Bessel function to compute the derivative.
    """
    x, nu = primals
    dx, dnu = tangents
    primal_out = mod_bessel(x, nu)
    bessel_derivative = -mod_bessel(x, nu+1) + (nu / x) * primal_out
    bessel_derivative = jnp.where(x == 0.0, 0.0, bessel_derivative)     # Handle x=0 case
    tangent_out = bessel_derivative * dx
    return primal_out, tangent_out

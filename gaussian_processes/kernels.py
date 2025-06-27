# This file contains all kernel classes for use in a Gaussian process regression model.  

import jax
import numpy as np
import jax.numpy as jnp
from jax.lax import cond
from jax.scipy.special import factorial, gamma
from abc import ABC, abstractmethod
from .support import is_positive_half_integer, mod_bessel
jax.config.update('jax_enable_x64', True)

class Kernel(ABC):
    """
    Abstract base class for all kernel classes. 
    """

    @abstractmethod
    def __call__(self, X1, X2=None):
        """        
        Evaluates the kernel function at inputs X. 

        Parameters
        ----------
        X1 : array-like of shape (n_points_1, ) or (n_points_1, n_features),
            List of first set of inputs; n_points_1 is number of points at which to 
            evaluate the kernel and n_features is the dimensionality of input space. 
        X2 : array-like of shape (n_points_2, ) or (n_points_2, n_features), default=None
            List of second set of inputs; n_features must match that of X1. If None, 
            uses X1. 
        """
        pass

    @abstractmethod
    def hyperparameters(self):
        """
        Returns a list of all hyperparameters.
        """
        pass

    @abstractmethod
    def get_params(self):
        """
        Returns the hyperparameter values. 
        """
        pass

    @abstractmethod
    def set_params(self, params):
        """
        Sets the kernel hyperparameter values. 

        Parameters
        ----------
        args : array-like of shape (n_hyperparameters, ) or float
            The hyperparameter values to set.
        """
        pass

    @abstractmethod
    def get_hyperprior(self):
        """
        Returns the prior distribution for the hyperparameters.
        """
        pass

    @abstractmethod
    def hyperprior(self):
        """
        Returns the prior distribution for the hyperparameters.
        """
        pass

    def __add__(self, kernel_2):
        """
        Adds this kernel to another kernel.
        
        Parameters
        ----------
        kernel_2 : Kernel
            The kernel to add to this kernel.
        """
        if not isinstance(kernel_2, Kernel):
            raise TypeError("Must pass a Kernel object.")
        return SumKernel(self, kernel_2)
    
    def __mul__(self, kernel_2):
        """
        Multiplies this kernel with another kernel.
        
        Parameters
        ----------
        kernel_2 : Kernel
            The kernel to multiply with this kernel.
        """
        if not isinstance(kernel_2, Kernel):
            raise TypeError("Must pass a Kernel object.")
        return ProductKernel(self, kernel_2)
    
    def __pow__(self, exponent):
        """
        Raises the kernel to some power.
        
        Parameters
        ----------
        exponent : int or float
            The power to raise this kernel to.
        """
        if not isinstance(exponent, (int, float)):
            raise TypeError("Exponent must be an int or float.")
        return ExponentialKernel(self, exponent)
    

class SumKernel(Kernel):
    """
    The sum of two kernels. 

    Attributes
    ----------
    kernel_1 : Kernel
        First kernel in the sum. 
    kernel_2 : Kernel
        Second kernel in the sum.
    """

    def __init__(self, kernel_1, kernel_2):
        """
        Initializes SumKernel object. 
        
        Parameters
        ----------
        kernel_1 : Kernel
            First kernel to add.
        kernel_2 : Kernel
            Second kernel to add.
        """
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
    
    def __call__(self, X1, X2=None):
        return self.kernel_1(X1, X2) + self.kernel_2(X1, X2)
    
    def hyperparameters(self):
        return self.kernel_1.hyperparameters() + self.kernel_2.hyperparameters()
    
    def get_params(self):
        return jnp.concatenate((self.kernel_1.get_params(), self.kernel_2.get_params()))
        
    def set_params(self, params):
        self.kernel_1.set_params(params[:len(self.kernel_1.hyperparameters())])
        self.kernel_2.set_params(params[len(self.kernel_1.hyperparameters()):])
    
    def get_hyperprior(self):
        return tuple(i+j for i, j in zip(self.kernel_1.get_hyperprior(), self.kernel_2.get_hyperprior()))

    def hyperprior(self):
        """
        Returns the prior distribution for the hyperparameters. Assumes that each hyperparameter is indipendent. 
        """
        return self.kernel_1.hyperprior() * self.kernel_2.hyperprior()


class ProductKernel(Kernel):
    """
    The product of two kernels. 

    Attributes
    ----------
    kernel_1 : Kernel
        First kernel in the product. 
    kernel_2 : Kernel
        Second kernel in the product.
    """

    def __init__(self, kernel_1, kernel_2):
        """
        Initializes ProductKernel object. 
        
        Parameters
        ----------
        kernel_1 : Kernel
            First kernel to multiply.
        kernel_2 : Kernel
            Second kernel to multiply.
        """
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
    
    def __call__(self, X1, X2=None):
        return self.kernel_1(X1, X2) * self.kernel_2(X1, X2)
    
    def hyperparameters(self):
        return self.kernel_1.hyperparameters() + self.kernel_2.hyperparameters()
    
    def get_params(self):
        return jnp.concatenate((self.kernel_1.get_params(), self.kernel_2.get_params()))
        
    def set_params(self, params):
        self.kernel_1.set_params(params[:len(self.kernel_1.hyperparameters())])
        self.kernel_2.set_params(params[len(self.kernel_1.hyperparameters()):])
    
    def get_hyperprior(self):
        return tuple(i+j for i, j in zip(self.kernel_1.get_hyperprior(), self.kernel_2.get_hyperprior()))

    def hyperprior(self):
        """
        Returns the prior distribution for the hyperparameters. Assumes that each hyperparameter is independent. 
        """
        return self.kernel_1.hyperprior() * self.kernel_2.hyperprior()


class ExponentialKernel(Kernel):
    """
    A kernel raised to some power. 

    Attributes
    ----------
    kernel : Kernel
        The kernel to raise to a power.
    exponent : int or float
        The exponent to raise the kernel to.
    """

    def __init__(self, kernel, exponent):
        """
        Initializes ExponentialKernel object. 
        
        Parameters
        ----------
        kernel : Kernel
            The kernel to raise to a power.
        exponent : int or float
            The exponent to raise the kernel to.
        """
        self.kernel = kernel
        if not isinstance(exponent, (int, float)):
            raise TypeError("Exponent must be an int or float.")
        self.power = exponent
    
    def __call__(self, X1, X2=None):
        return self.kernel(X1, X2)**self.power
    
    def hyperparameters(self):
        return self.kernel.hyperparameters()
    
    def get_params(self):
        return self.kernel.get_params()
        
    def set_params(self, params):
        self.kernel.set_params(params)
    
    def get_hyperprior(self):
        return self.kernel.get_hyperprior()
    
    def hyperprior(self):
        return self.kernel.hyperprior()


class IndividualKernel(Kernel):
    """
    Abstract base class for individual kernels. 
    Inherits from Kernel class.

    Attributes
    ----------
    theta : array-like
        Array-like containing the log kernel hyperparameter value.
    prior_type : str
        Type of prior distribution for the hyperparameter.
    prior_params : array-like
        Parameters for the prior distribution.
    hyperparameter_types : list
        List of hyperparameter types (i.e. length scale).
    """


    def __init__(self, hyperparameter, prior_type='uniform', prior_params=None, hyperparameter_type='Hyperparameter Type'):
        """
        Initializes IndividualKernel object. 

        Parameters
        ----------
        hyperparameter : int or float
            Initial hyperparameter of the kernel.
        prior_type : str, default='uniform'
            Type of prior distribution for the hyperparameter.
            Supported values are 'uniform', 'log_normal', and 'fixed'.
        prior_params : array-like, default=None
            Parameters for the prior distribution.
            For uniform prior, provide the bounds for the allowed range. 
            For log-normal prior, provide the desired median and standard deviation.
            The standard deviation is in log space. 
            For a fixed prior (the hyperparameter is fixed), the prior parameter 
            is set to the hyperparameter value.
        hyperparameter_type : str, default='Hyperparameter Type'
            Type of hyperparameter. Different for each child class. 
        """
        if not isinstance(hyperparameter, (int, float)):
            raise TypeError('hyperparameter must be int or float. ')
        self.theta = jnp.log(jnp.array([hyperparameter], dtype=float))
        if prior_type != 'uniform' and prior_type != 'log_normal' and prior_type != 'fixed':
            raise ValueError('prior_type must be either "uniform", "log_normal" or "fixed".')
        self.prior_type = prior_type
        if prior_params is not None:
            self.prior_params = jnp.array(prior_params, dtype=float)
        elif prior_type == 'log_normal':
            self.prior_params = jnp.array([1., 1.])                            # Mean and std for log-normal prior
        elif prior_type == 'uniform':
            self.prior_params = jnp.array([1e-3, 1e3], dtype=float)            # Bounds for uniform prior
        else:
            self.prior_params = jnp.array([hyperparameter])                    # Fixed prior, set to hyperparameter value   
        self.hyperparameter_type = [hyperparameter_type]

    def __call__(self, X1, X2=None):
        pass

    def hyperparameters(self):
        return self.hyperparameter_type

    def get_params(self):
        return jnp.exp(self.theta)
    
    def set_params(self, params):
        if self.prior_type != 'fixed':
            self.theta = jnp.log(jnp.array(params, dtype=float))
    
    def get_hyperprior(self):
        return [self.prior_type], [self.prior_params]

    @abstractmethod
    def hyperprior(self):
        """
        Returns the prior distribution for the hyperparameter.
        """
        # Log-normal prior
        if self.prior_type == 'log_normal':
            # Calculate mean in log space
            mu = jnp.log(self.prior_params[0]) + self.prior_params[1]**2 
            return 1/(jnp.sqrt(2*jnp.pi)*jnp.exp(self.theta[0])*self.prior_params[1]) * \
                jnp.exp(-0.5*((self.theta[0] - mu)/self.prior_params[1])**2)
        
        # Uniform prior
        elif self.prior_type == 'uniform': 
            in_bounds = jnp.logical_and(
                jnp.greater_equal(jnp.exp(self.theta[0]), self.prior_params[0]),
                jnp.less_equal(jnp.exp(self.theta[0]), self.prior_params[1])
            )
            uniform_density = 1/(self.prior_params[1] - self.prior_params[0])
            return cond(in_bounds, lambda: uniform_density, lambda: 0.0)

        # Fixed prior
        else:
            return 1.0              # Return 1 since log(1) = 0, nothing is added to the MAP estimator
    

class SE(IndividualKernel):
    """
    Squared exponential kernel. 

    Attributes
    ----------
    theta : array-like
        Array-like containing the log kernel hyperparameter value.
    prior_type : str
        Type of prior distribution for the hyperparameter.
    prior_params : array-like
        Parameters for the prior distribution.
    hyperparameter_types : list
        List containing the type of hyperparameter, here the characteristic length scale.
    """

    def __init__(self, length_scale=1.0, prior_type='uniform', prior_params=None):
        """
        Initializes SE kernel object. 

        Parameters
        ----------
        length_scale : int or float, defualt=1.0
            Initial length scale hyperparameter value.
        prior_type : str, default='uniform'
            Type of prior distribution for the hyperparameter.
            Supported values are 'uniform', 'log_normal', and 'fixed'.
        prior_params : array-like, default=None
            Parameters for the prior distribution.
            For uniform prior, provide the bounds for the allowed range. 
            For log-normal prior, provide the desired median and standard deviation.
            The standard deviation is in log space. 
            For a fixed prior (the hyperparameter is fixed), the prior parameter 
            is set to the hyperparameter value.
        """
        super().__init__(length_scale, prior_type, prior_params, 'RBF Length Scale')
    
    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        if isinstance(X1, (float, int)):
            squared_distance = (X1 - X2)**2
        else:
            squared_distance = (X1[:, None] - X2[None, :])**2
        return jnp.exp(-squared_distance/(2*jnp.exp(self.theta[0])**2))
    
    def hyperparameters(self):
        return super().hyperparameters()
    
    def get_params(self):
        return super().get_params()
    
    def set_params(self, params):
        super().set_params(params)
    
    def get_hyperprior(self):
        return super().get_hyperprior()

    def hyperprior(self):
        return super().hyperprior()


class Dot(IndividualKernel):
    """
    Dot Product kernel. 

    Attributes
    ----------
    theta : array-like
        Array-like containing the log kernel hyperparameter value.
    prior_type : str
        Type of prior distribution for the hyperparameter.
    prior_params : array-like
        Parameters for the prior distribution.
    hyperparameter_types : list
        List containing the type of hyperparameter, here the inhomogeneity constant.
    """

    def __init__(self, sigma_0=1.0, prior_type='uniform', prior_params=None):
        """
        Initializes Dot Product kernel object. 
        
        Parameters
        ----------
        sigma_0 : int or float, default=1.0
            Inhomogeneity of the kernel.
        prior_type : str, default='uniform'
            Type of prior distribution for the hyperparameter.
            Supported values are 'uniform', 'log_normal', and 'fixed'.
        prior_params : array-like, default=None
            Parameters for the prior distribution.
            For uniform prior, provide the bounds for the allowed range. 
            For log-normal prior, provide the desired median and standard deviation.
            The standard deviation is in log space. 
            For a fixed prior (the hyperparameter is fixed), the prior parameter 
            is set to the hyperparameter value.
        """
        super().__init__(sigma_0, prior_type, prior_params, 'Linear Inhomogeneity')

    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        if isinstance(X1, (float, int)):
            product = X1 * X2
        else:
            product = X1[:, None] * X2[None, :]
        return jnp.exp(self.theta[0])**2 + product
    
    def hyperparameters(self):
        return super().hyperparameters()
    
    def get_params(self):
        return super().get_params()
    
    def set_params(self, params):
        super().set_params(params)
    
    def get_hyperprior(self):
        return super().get_hyperprior()

    def hyperprior(self):
        return super().hyperprior()


class Constant(IndividualKernel):
    """
    Constant kernel. 

    Attributes
    ----------
    theta : array-like
        Array-like containing the log kernel hyperparameter value.
    prior_type : str
        Type of prior distribution for the hyperparameter.
    prior_params : array-like
        Parameters for the prior distribution.
    hyperparameter_types : list
        List containing the type of hyperparameter, here the constant.
    """

    def __init__(self, sigma_0=1.0, prior_type='uniform', prior_params=None):
        """
        Initializes Constant kernel object. 
        
        Parameters
        ----------
        sigma_0 : int or float, default=1.0
            Constant value of the kernel.
        prior_type : str, default='uniform'
            Type of prior distribution for the hyperparameter.
            Supported values are 'uniform', 'log_normal', and 'fixed'.
        prior_params : array-like, default=None
            Parameters for the prior distribution.
            For uniform prior, provide the bounds for the allowed range. 
            For log-normal prior, provide the desired median and standard deviation.
            The standard deviation is in log space. 
            For a fixed prior (the hyperparameter is fixed), the prior parameter 
            is set to the hyperparameter value.
        """
        super().__init__(sigma_0, prior_type, prior_params, 'Constant Value')

    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        if isinstance(X1, (float, int)):
            return jnp.exp(self.theta[0])
        else:
            return jnp.exp(self.theta[0]) * jnp.ones((X1.shape[0], X2.shape[0]))
    
    def hyperparameters(self):
        return super().hyperparameters()
    
    def get_params(self):
        return super().get_params()
    
    def set_params(self, params):
        super().set_params(params)
    
    def get_hyperprior(self):
        return super().get_hyperprior()
    
    def hyperprior(self):
        return super().hyperprior()


class Matern(IndividualKernel):
    """
    Matern kernel. 

    Attributes
    ----------
    theta : array-like
        Array-like containing the log kernel hyperparameter value.
    nu : float
        Parameter controlling the smoothness of predicted functions.
    prior_type : str
        Type of prior distribution for the hyperparameter.
    prior_params : array-like
        Parameters for the prior distribution.
    hyperparameter_types : list
        List containing the type of hyperparameter, here the characteristic length scale.
    """

    def __init__(self, length_scale=1.0, nu=1.5, prior_type='uniform', prior_params=None):
        """
        Initializes Matern kernel object. 
        
        Parameters
        ----------
        length_scale : int or float, default=1.0
            Characteristic length scale of the kernel.
        nu : float, default=1.5
            Prameter controlling the smoothness of predicted functions. 
            Higher nu gives a more smooth prediction.
            Must be a positive value. 
        prior_type : str, default='uniform'
            Type of prior distribution for the hyperparameter.
            Supported values are 'uniform', 'log_normal', and 'fixed'.
        prior_params : array-like, default=None
            Parameters for the prior distribution.
            For uniform prior, provide the bounds for the allowed range. 
            For log-normal prior, provide the desired median and standard deviation.
            The standard deviation is in log space. 
            For a fixed prior (the hyperparameter is fixed), the prior parameter 
            is set to the hyperparameter value.
        """
        if is_positive_half_integer(nu):
            self.half_integer = True
        else:
            self.half_integer = False
        if not isinstance(nu, (float, int)) or nu <= 0:
            raise ValueError('Parameter nu must be a positive value.')
        self.nu = nu
        super().__init__(length_scale, prior_type, prior_params, 'Matern Length Scale')
    
    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        if isinstance(X1, (float, int)):
            distance = jnp.abs(X1 - X2)
        else:
            distance = jnp.abs(X1[:, None] - X2[None, :])
        if self.half_integer:
            return self.half_integer_nu(distance)
        else:
            return self.general_nu(distance)

    
    def half_integer_nu(self, distance):
        """
        Calculates the value of the kernel for a half-integer nu. 
        Implements Eqn. 4.16 from Rasmussen and Williams.

        Parameters
        ----------
        distance : array-like of shape (n_points_1, n_points_2)
            The distance between the points at which to evaluate the kernel.
            Can be a single value or an array of values.

        Returns
        -------
        array-like of shape (n_points_1, n_points_2)
            The value of the kernel for the given inputs. 
        """
        p = int(self.nu - 0.5)
        sum = 0
        for i in range(p+1):
            binomial = factorial(p+i) / (factorial(i) * factorial(p-i))
            power = (jnp.sqrt(8*self.nu) * distance / jnp.exp(self.theta[0]))**(p-i)
            sum += binomial * power
        prefactor = jnp.exp(-jnp.sqrt(2*self.nu) * distance / jnp.exp(self.theta[0])) * gamma(p+1) / gamma(2*p+1)
        return prefactor * sum
    
    def general_nu(self, distance):
        """
        Calculates the value of the kernel for a general nu. 
        Implements Eqn. 4.14 from Rasmussen and Williams.

        Parameters
        ----------
        distance : array-like of shape (n_points_1, n_points_2)
            The distance between the points at which to evaluate the kernel.
            Can be a single value or an array of values.

        Returns
        -------
        array-like of shape (n_points_1, n_points_2)
            The value of the kernel for the given inputs. 
        """
        arg = jnp.sqrt(2*self.nu) * distance / jnp.exp(self.theta[0])
        # Avoid division by zero during gradient descent; this value is set to the limiting value below. 
        arg = jnp.where(arg == 0, 1.0, arg)
        return_arr = 2**(1-self.nu) / gamma(self.nu) * arg**self.nu * mod_bessel(arg, self.nu)
        # Handle the case where distance is zero, limiting value is 1
        return jnp.where(distance == 0, 1.0, return_arr)
    
    def hyperparameters(self):
        return super().hyperparameters()
    
    def get_params(self):
        return super().get_params()
    
    def set_params(self, params):
        super().set_params(params)
    
    def get_hyperprior(self):
        return super().get_hyperprior()

    def hyperprior(self):
        return super().hyperprior()
    

class RationalQuadratic(IndividualKernel):
    """
    Rational Quadratic kernel. 

    Attributes
    ----------
    theta : array-like
        Array-like containing the log kernel hyperparameter value.
    alpha : float
        Scale mixture parameter. 
    prior_type : str
        Type of prior distribution for the hyperparameter.
    prior_params : array-like
        Parameters for the prior distribution.
    hyperparameter_types : list
        List containing the type of hyperparameter, here the characteristic length scale.
    """

    def __init__(self, length_scale=1.0, alpha=1.0, prior_type='uniform', prior_params=None):
        """
        Initializes Rational Quadratic kernel object. 
        
        Parameters
        ----------
        length_scale : int or float, default=1.0
            Characteristic length scale of the kernel.
        alpha : int or float, default=1.0
            Scale mixture parameter. 
            Higher alpha gives a more smooth prediction.
            Must be positive. 
        prior_type : str, default='uniform'
            Type of prior distribution for the hyperparameter.
            Supported values are 'uniform', 'log_normal', and 'fixed'.
        prior_params : array-like, default=None
            Parameters for the prior distribution.
            For uniform prior, provide the bounds for the allowed range. 
            For log-normal prior, provide the desired median and standard deviation.
            The standard deviation is in log space. 
            For a fixed prior (the hyperparameter is fixed), the prior parameter 
            is set to the hyperparameter value.
        """
        if not isinstance(alpha, (float, int)) or alpha < 0:
            raise ValueError('Scale mixture parameter alpha must positive.')
        self.alpha = alpha
        super().__init__(length_scale, prior_type, prior_params, 'Rational Quadratic Length Scale')
    
    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        if isinstance(X1, (float, int)):
            squared_distance = (X1 - X2)**2
        else:
            squared_distance = (X1[:, None] - X2[None, :])**2
        # Implement Eqn. 4.19 from Rasmussen and Williams
        return (1 + squared_distance / (2*self.alpha*jnp.exp(self.theta[0])**2))**(-self.alpha)
    
    def hyperparameters(self):
        return super().hyperparameters()
    
    def get_params(self):
        return super().get_params()
    
    def set_params(self, params):
        super().set_params(params)
    
    def get_hyperprior(self):
        return super().get_hyperprior()

    def hyperprior(self):
        return super().hyperprior()
    

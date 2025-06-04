# This file contains optimizers for use in the Gaussian process regression model.

import jax
import jax.example_libraries.optimizers
import jax.numpy as jnp
import numpy as np
import math
from jax.scipy.optimize import minimize
from abc import ABC, abstractmethod
import math
import warnings
from . import kernels
jax.config.update('jax_enable_x64', True)


class Optimizer(ABC):
    """
    Abstract base class for all optimizers. 
    """
    
    def __init__(self, kernel, train_X, train_y, cov_mat, points):
        """
        Initializes the optimizer with a Gaussian process model.
        
        Parameters
        ----------
        kernel : Kernel
            The kernel to optimize.
        train_X : array-like of shape (n_samples, ) or (n_samples, n_features)
            Training sample features.
        train_y : array-like of shape (n_samples, )
            Training sample values.
        cov_mat : array-like of shape (n_samples, n_samples)
            Covariance matrix for the training data in the Gaussian process model.
        points : array-like of shape (n_sets, ), default=None
            Number of points in each set of training data.
        """
        if not isinstance(kernel, kernels.Kernel):
            raise TypeError("kernel must be an instance of the Kernel class.")
        self.kernel = kernel
        self.train_X = train_X
        self.train_y = train_y
        self.cov_mat = cov_mat
        self.points = points
    

    @abstractmethod
    def optimize(self):
        """
        Uses BFGS algorithm to find the optimal kernel hyperparameters.
        Begins at the initial hyperparameters of the kernel.

        Returns
        -------
        theta : array-like of shape (n_hyperparameters, )
            The optimized hyperparameters.
        loss : float
            Final value of the loss function for the optimized hyperparameters. 
        """
        start_hyper = jnp.log(self.kernel.get_params())
        fitted = minimize(self.loss, start_hyper, method='BFGS')
        theta = fitted.x
        loss = self.loss(theta)
        return jnp.exp(theta), loss


    @abstractmethod
    def optimize_restarts(self):
        """
        Uses BFGS algorithm to find the optimal kernel hyperparameters.
        Loops over a grid of hyperparameters to find the starting point that gives the 
        smallest loss, reducing the chance of finding a local minimum.
        
        Returns
        -------
        theta : array-like of shape (n_hyperparameters, )
            The optimized hyperparameters.
        loss : float
            Final value of the loss function for the optimized hyperparameters. 
        """
        # Initialize hyperparameters
        hyper_types, hyper_priors = self.kernel.get_hyperprior()

        start_hyper = []
        for i in range(len(hyper_types)):
            # For log-normal priors, start at median, +/-1,2,3 sigma for each hyperparameter
            if hyper_types[i] == 'log_uniform':
                start_hyper.append(np.array([hyper_priors[i][0]*jnp.exp(j*hyper_priors[i][1]) for j in range(-3,4)]))

            # For uniform priors, start at each order of magnitude
            elif hyper_types[i] == 'uniform':
                log_bounds = math.ceil(jnp.log10(hyper_priors[i][0])), math.floor(jnp.log10(hyper_priors[i][1]))
                orders = np.arange(log_bounds[0], log_bounds[1]+1, dtype=float)
                start_hyper.append(np.array([10**order for order in orders]))
            
            # For fixed hyperparameters, only start at the fixed value
            else:
                start_hyper.append(np.array(hyper_priors[i]))

        # Create a grid of all combinations of starting hyperparameters
        start_hyper = jnp.log(jnp.array(jnp.meshgrid(*start_hyper, indexing='ij')).T.reshape(-1, len(hyper_types)))
        theta = jnp.log(self.kernel.get_params())
        min_loss = jnp.inf
        # Loop over all combinations of starting hyperparameters, keeping the optimized hyperparameters with the smallest loss
        for start in start_hyper:
            fitted = minimize(self.loss, start, method='BFGS')
            loss = self.loss(fitted.x)
            if loss < min_loss:
                min_loss = loss
                theta = fitted.x

        return jnp.exp(theta), min_loss
    

    @abstractmethod
    def adam(self, tolerance=1e-3, step_size=1e-2, b1=0.9, b2=0.999):
        """
        Uses Adam to find the optimal kernel hyperparameters.
        Begins at the initial hyperparameters of the kernel.

        Parameters
        ----------
        tolerance : float, default=1e-3
            Threshold for convergence for Adam optimizer.
        step_size : float, default=1e-2
            Step size for Adam optimizer.
        b1 : float, default=0.9
            Exponential decay rate for first moment estimates.
        b2 : float, default=0.999
            Exponential decay rate for second moment estimates. 
        
        Returns
        -------
        theta : array-like of shape (n_hyperparameters, )
            The optimized hyperparameters.
        loss : float
            Final value of the loss function for the optimized hyperparameters. 
        """
        init_fn, update_fn, params_fn = jax.example_libraries.optimizers.adam(step_size, b1, b2)
        params = jnp.log(self.kernel.get_params())
        
        opt_state = init_fn(params)
        loss_and_grad = jax.value_and_grad(self.loss)
        prev_loss = 0
        update = np.inf
        step = 1
        while update > tolerance:
            params = params_fn(opt_state)
            loss, grads = loss_and_grad(params)
            if math.isnan(loss):
                warnings.warn("Selected hyperparameters are not returning a positive-definite covariance matrix.")
            if loss == jnp.inf:
                warnings.warn("Optimal hyperparameters are outside the given bounds.")
            opt_state = update_fn(step, grads, opt_state)
            update = jnp.abs(loss-prev_loss)
            prev_loss = loss
            if step % 50 == 0:
                print('')
                print(loss,update)
                print(jnp.exp(params_fn(opt_state)))
            step += 1
        theta = params_fn(opt_state)
        return jnp.exp(theta), loss


    @abstractmethod
    def loss(self, theta):
        pass


class MarginalLikelihoodOptimizer(Optimizer):
    """
    Chooses the hyperparameters of the kernel of a Gaussian process model by maximizing the 
    marginal likelihood with gradient descent. 
    
    Attributes
    ----------
    kernel : Kernel
        The kernel to optimize.
    train_X : array-like of shape (n_samples, ) or (n_samples, n_features)
        Training sample features.
    cov_mat : array-like of shape (n_samples, n_samples)
        Covariance matrix for the training data in the Gaussian process model.
    """
    
    def __init__(self, kernel, train_X, train_y, cov_mat, points=None):
        """
        Initializes the optimizer with a Gaussian process model.
        
        Parameters
        ----------
        kernel : Kernel
            The kernel to optimize.
        train_X : array-like of shape (n_samples, ) or (n_samples, n_features)
            Training sample features.
        train_y : array-like of shape (n_samples, )
            Training sample values.
        cov_mat : array-like of shape (n_samples, n_samples)
            Covariance matrix for the training data.
        points : array-like of shape (n_sets, ), default=None
            Number of points in each set of training data. Not used for this optimizer.
        """
        super().__init__(kernel, train_X, train_y, cov_mat, points)

    
    def optimize(self): 
        """
        Uses BFGS algorithm to find the optimal kernel hyperparameters.
        Begins at the initial hyperparameters of the kernel.

        Returns
        -------
        theta : array-like of shape (n_hyperparameters, )
            The optimized hyperparameters.
        loss : float
            Final value of the loss function for the optimized hyperparameters. 
        """
        return super().optimize()


    def optimize_restarts(self):
        """
        Uses BFGS algorithm to find the optimal kernel hyperparameters.
        Loops over a grid of hyperparameters to find the starting point that gives the 
        smallest loss, reducing the chance of finding a local minimum.
        
        Returns
        -------
        theta : array-like of shape (n_hyperparameters, )
            The optimized hyperparameters.
        loss : float
            Final value of the loss function for the optimized hyperparameters. 
        """
        return super().optimize_restarts()
    

    def adam(self, tolerance=1e-3, step_size=1e-2, b1=0.9, b2=0.999):
        """
        Uses Adam to find the optimal kernel hyperparameters.
        Begins at the initial hyperparameters of the kernel.

        Parameters
        ----------
        tolerance : float, default=1e-3
            Threshold for convergence for Adam optimizer.
        step_size : float, default=1e-2
            Step size for Adam optimizer.
        b1 : float, default=0.9
            Exponential decay rate for first moment estimates.
        b2 : float, default=0.999
            Exponential decay rate for second moment estimates. 

        Returns
        -------
        theta : array-like of shape (n_hyperparameters, )
            The optimized hyperparameters.
        loss : float
            Final value of the loss function for the optimized hyperparameters. 
        """
        return super().adam(tolerance, step_size, b1, b2)

    
    def loss(self, theta):
        """
        Objective function to minimize.
        
        Parameters
        ----------
        theta : array-like of shape (n_hyperparameters, )
            The log hyperparameters to optimize.
        
        Returns
        -------
        loss : float
            The negative log marginal likelihood.
        """
        params = jnp.exp(theta)
        self.kernel.set_params(params)
        K11 = self.kernel(self.train_X) + self.cov_mat
        L = jnp.linalg.cholesky(K11)
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, self.train_y))
        marginal_likelihood = -0.5 * jnp.linalg.slogdet(K11)[1] - 0.5 * jnp.dot(self.train_y.T, alpha) 
        hyperprior = jnp.log(self.kernel.hyperprior())
        return -marginal_likelihood - hyperprior


class LOOOptimizer(Optimizer):
    """
    Chooses the hyperparameters of the kernel of a Gaussian process model by maximizing the 
    marginal likelihood with gradient descent. 
    
    Attributes
    ----------
    kernel : Kernel
        The kernel to optimize.
    train_X : array-like of shape (n_samples, ) or (n_samples, n_features)
        Training sample features.
    cov_mat : array-like of shape (n_samples, n_samples)
        Covariance matrix for the training data in the Gaussian process model.
    points : array-like of shape (n_sets, )
        Number of points in each set of training data.
    """
    
    def __init__(self, kernel, train_X, train_y, cov_mat, points=None):
        """
        Initializes the optimizer with a Gaussian process model.
        
        Parameters
        ----------
        kernel : Kernel
            The kernel to optimize.
        train_X : array-like of shape (n_samples, ) or (n_samples, n_features)
            Training sample features.
        cov_mat : array-like of shape (n_samples, n_samples)
            Covariance matrix for the training data.
        points : array-like of shape (n_sets, ), default=None
            Number of points in each set of training data.
        """
        super().__init__(kernel, train_X, train_y, cov_mat, np.array(points))
    
    
    def optimize(self):
        """
        Uses BFGS algorithm to find the optimal kernel hyperparameters.
        Begins at the initial hyperparameters of the kernel.

        Returns
        -------
        theta : array-like of shape (n_hyperparameters, )
            The optimized hyperparameters.
        loss : float
            Final value of the loss function for the optimized hyperparameters. 
        """
        return super().optimize()
    

    def optimize_restarts(self):
        """
        Uses BFGS algorithm to find the optimal kernel hyperparameters.
        Loops over a grid of hyperparameters to find the starting point that gives the 
        smallest loss, reducing the chance of finding a local minimum.
        
        Returns
        -------
        theta : array-like of shape (n_hyperparameters, )
            The optimized hyperparameters.
        loss : float
            Final value of the loss function for the optimized hyperparameters. 
        """
        return super().optimize_restarts()
    

    def adam(self, tolerance=1e-3, step_size=1e-2, b1=0.9, b2=0.999):
        """
        Uses Adam to find the optimal kernel hyperparameters.
        Begins at the initial hyperparameters of the kernel.

        Parameters
        ----------
        tolerance : float, default=1e-3
            Threshold for convergence for Adam optimizer.
        step_size : float, default=1e-2
            Step size for Adam optimizer.
        b1 : float, default=0.9
            Exponential decay rate for first moment estimates.
        b2 : float, default=0.999
            Exponential decay rate for second moment estimates. 
        
        Returns
        -------
        theta : array-like of shape (n_hyperparameters, )
            The optimized hyperparameters.
        loss : float
            Final value of the loss function for the optimized hyperparameters. 
        """
        return super().adam(tolerance, step_size, b1, b2)
    
        
    def loss(self, theta):
        """
        Objective function to minimize.
        
        Parameters
        ----------
        theta : array-like of shape (n_hyperparameters, )
            The hyperparameters to optimize.
        
        Returns
        -------
        loss : float
            The negative log marginal likelihood.
        """
        params = jnp.exp(theta)
        self.kernel.set_params(params)
        K11 = self.kernel(self.train_X) + self.cov_mat
        loss = 0
        index = 0

        # Implement Eqn. 5.10, 5.12 from Rasmussen and Williams
        K11inv = jnp.linalg.inv(K11)
        pred_variance = jnp.diag(K11inv)
        L = jnp.linalg.cholesky(K11)
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, self.train_y))

        # Loop through each dataset, calculating the loss for each
        for i in range(len(self.points)):
            indices = np.arange(self.points[i]) + index
            # y_i cancels in 5.10, so omit in definition of mu
            mu = alpha[indices] / pred_variance[indices]
            sigma2 = pred_variance[indices]
            # The loss for each point is normalized by the uncertainty of each point
            #dataset_loss = jnp.sum(1/self.cov_mat[indices, indices] * -0.5 * (jnp.log(sigma2) + mu**2/sigma2 + jnp.log(2*jnp.pi)))
            dataset_loss = jnp.sum(-0.5 * (jnp.log(sigma2) + mu**2/sigma2))
            # Update the index for the next dataset
            index += self.points[i]
            # The loss for each dataset is normalized by the number of points in each dataset
            loss += dataset_loss / self.points[i]

        # Add the hyperprior to the loss
        hyperprior = jnp.log(self.kernel.hyperprior())

        return -loss - hyperprior
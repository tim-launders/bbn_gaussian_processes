# This file contains the main class for the Gaussian process regression model.

import jax
import numpy as np
import jax.numpy as jnp
from . import kernels
from . import optimizers
from jax.scipy.linalg import block_diag
from jax.random import PRNGKey, split, multivariate_normal
from jax.lax.linalg import cholesky
jax.config.update('jax_enable_x64', True)


class GPR:
    """
    Gaussian Process Regression (GPR) class. 

    Attributes
    ----------
    kernel : Kernel
        The kernel to use for the GPR model.
    noise : float
        The noise level in the observations. 
    optimizer_type : str
        The optimizer to use for hyperparameter optimization.
    X_train : array-like of shape (n_samples, ) or (n_samples, n_features)
        Training input samples.
    y_train : array-like of shape (n_samples, )
        Training output values.
    cov_mat : array-like of shape (n_samples, n_samples)
        Covariance matrix for the training data.
    """


    def __init__(self, kernel, noise=1e-10, optimizer_type=None, optimize_restarts=False):
        """
        Initializes the GPR model with a specified kernel and noise level. 

        Parameters
        ----------
        kernel : Kernel
            The kernel to use for the GPR model. 
        noise : float, default=1e-10
            The noise level in the observations. Assists with numerical stability.
        optimizer_type : str, default=None
            The optimizer to use for hyperparameter optimization. If None, no 
            optimization is performed. Allowed optimizers are 'marginal_likelihood' and 'LOO'.
        optimize_restarts : bool, default=False
            Determines whether to use the initial values passed in for the kernel hyperparameters 
            to begin hyperparameter optimization or to explore different starting points in 
            hyperparameter space. 
        """
        if not isinstance(kernel, kernels.Kernel):
            raise TypeError("kernel must be an instance of the Kernel class.")
        self.kernel = kernel
        self.noise = noise
        if optimizer_type is not None:
            if optimizer_type != "marginal_likelihood" and optimizer_type != "LOO":
                raise ValueError("Invalid optimizer type. Must be 'marginal_likelihood' or 'LOO'.")
        self.optimizer_type = optimizer_type
        self.optimize_restarts = optimize_restarts


    def fit(self, X, y, total_error=None, normalization_error=None, points=None):
        """
        Fits the GPR model to the training data. 

        Parameters
        ----------
        X : array-like of shape (n_samples, ) or (n_samples, n_features)
            Training input samples. 
        y : array-like of shape (n_samples, )
            Training output values. 
        total_error : array-like of shape (n_samples, ), default=None
            Total error for each training sample.
        normalization_error : array-like of shape (n_samples, ), default=None
            Normalization error for each training sample. If total_error is not 
            None and normalization_error is None, it is set to zero.
        points : array-like of shape (n_sets, ), default=None
            Number of points in each set of training data. If total_error is not 
            None and points is None, it is set to a (1, ) array with value equal to 
            the number of training samples. 
        
        Returns
        -------
        params : array-like of shape (n_hyperparameters, )
            Hyperparameters of the kernel after fitting. If no optimization is 
            performed, returns the original hyperparameters. 
        loss : float
            Loss for the trained kernel. If no optimization is performed, returns 0.
        """
        self.X_train = jnp.array(X)
        self.y_train = jnp.array(y)
        
        if total_error is not None:
            if len(total_error) != len(X):
                raise ValueError("Number of total errors and samples must match.")
            if normalization_error is None: 
                normalization_error = jnp.zeros(len(total_error))
            elif len(normalization_error) != len(X):
                raise ValueError("Number of normalization errors and samples much match.")
            if points is None: 
                points = len(total_error) * jnp.ones(1)
            elif np.sum(np.array(points)) != len(X):
                raise ValueError("Number of points and samples must match.")
            self.cov_mat = self.calculate_covariance(total_error, normalization_error, points)

        else:
            self.cov_mat = jnp.zeros((len(X), len(X)))

        # If optimizer is given, optimize the kernel hyperparameters using the optimizer specified
        if self.optimizer_type is not None:
            if self.optimizer_type == "marginal_likelihood":
                optimizer = optimizers.MarginalLikelihoodOptimizer(self.kernel, self.X_train, self.y_train, self.cov_mat)
            elif self.optimizer_type == "LOO":
                optimizer = optimizers.LOOOptimizer(self.kernel, self.X_train, self.y_train, self.cov_mat, points)
            if self.optimize_restarts == True:
                params, loss = optimizer.optimize_restarts()
            else:
                params, loss = optimizer.optimize()
            self.kernel.set_params(params)
            return params, loss
        else:
            return self.get_kernel_hyperparameters(), 0.0
        

    def predict(self, X_pred, return_std=False, return_cov=False):
        """
        Predicts the output for the given input samples. 

        Parameters
        ----------
        X_pred : array-like of shape (n_samples, ) or (n_samples, n_features)
            Inputs at which to predict the output.
        return_std : bool, default=False
            If True, returns the standard deviation of the predictions.
        return_cov : bool, default=False
            If True, returns the covariance matrix of the predictions.
        
        Returns
        -------
        y_pred : array-like of shape (n_samples, )
            Mean of conditional distribution.
        y_std : array-like of shape (n_samples, )
            Standard deviation of conditional distribution.
            Only returned if return_std=True.
        y_cov : array-like of shape (n_samples, n_samples)
            Full covariance matrix of conditional distribution.
            Only returned if return_cov=True. 
        """
        K11 = self.kernel(self.X_train) + self.cov_mat + self.noise * jnp.eye(len(self.X_train))

        # Implement Algorithm 2.1 from "Gaussian Processes for Machine Learning" by Rasmussen and Williams
        L = cholesky(K11)
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, self.y_train))
        K12 = self.kernel(self.X_train, X_pred)
        y_pred = K12.T @ alpha                           # Eqn. 2.25
        v = jnp.linalg.solve(L, K12)
        y_cov = self.kernel(X_pred) - v.T @ v            # Eqn. 2.26
        y_std = jnp.sqrt(jnp.diag(y_cov))
        
        if return_std:
            if return_cov:
                return y_pred, y_std, y_cov
            else:
                return y_pred, y_std
        elif return_cov:
            return y_pred, y_cov
        else:
            return y_pred
    
    
    def draw_samples(self, X_pred, n_draws=1, seed=0): 
        """
        Draws samples from the GP posterior.

        Parameters
        ----------
        X_pred : array-like of shape (n_samples, ) or (n_samples, n_features)
            Inputs at which to predict the output.
        n_draws : int, default=1
            Number of samples to draw.
        seed : int, default=0
            Seed to use for the key. 
        
        Returns
        -------
        samples : array-like of shape (n_draws, n_samples)
        """
        pred_mean, pred_cov = self.predict(X_pred, return_cov=True)

        key = PRNGKey(seed)
        key, subkey = split(key)
        samples = multivariate_normal(subkey, pred_mean, pred_cov, shape=(n_draws,), method="svd")
        return samples


    def calculate_covariance(self, total_error, normalization_error, points):
        """
        Calculates the covariance matrix before adding the kernel for the training data. 

        Parameters
        ----------
        total_error : array-like of shape (n_samples, )
            Total error for each training sample.
        normalization_error : array-like of shape (n_samples, )
            Normalization error for each training sample.
        points : array-like of shape (n_sets, )
            Number of points in each set of training data.

        Returns
        -------
        cov_mat : array-like of shape (n_samples, n_samples)
            Covariance matrix for the training data. 
        """
        num_sets = len(points)

        cov_mat = jnp.zeros((0,0))
        ind = 0

        for i in range(num_sets):
            pts = int(points[i])
            diagonal_elts = jnp.diag(total_error[ind:ind+pts]**2)
            normalizations = jnp.outer(normalization_error[ind:ind+pts], normalization_error[ind:ind+pts])
            # Zero out the diagonal elements of the normalization matrix, as that is included in the total error
            normalizations = normalizations - jnp.diag(jnp.diag(normalizations))    
            this_set = diagonal_elts + normalizations
            cov_mat = block_diag(cov_mat, this_set)
            ind += pts

        return cov_mat
        
    
    def get_kernel_hyperparameters(self):
        """
        Returns the hyperparameters of the kernel. 

        Returns
        -------
        hyper : array-like of shape (n_hyperparameters, )
            List of hyperparameter values of the kernel. 
        """
        hyper = self.kernel.get_params()
        return hyper
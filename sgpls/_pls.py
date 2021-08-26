import numpy as np

from abc import ABCMeta, abstractmethod

from sklearn.base import RegressorMixin, MultiOutputMixin
from sklearn.cross_decomposition._pls import \
    _get_first_singular_vectors_power_method

from ._base import _PLSBase
from ._base import svd_cross_product

__all__ = ['PLSCanonical', 'PLSRegression']


def _pls_inner_loop(X, Y, algorithm, max_iter=500, tol=1e-06,
                    norm_y_weights=True):
    """Inner loop for PLS weights estimation
    """
    if algorithm == "nipals":
        # Replace columns that are all close to zero with zeros
        Y_mask = np.all(np.abs(Y) < 10 * np.finfo(float).eps, axis=0)
        Y[:, Y_mask] = 0.0
    
        u, v, n_iter = \
            _get_first_singular_vectors_power_method(
                X, Y, mode="A", max_iter=max_iter,
                tol=tol, norm_y_weights=norm_y_weights)
    
    elif algorithm == "svd":
        # SVD returns PLS weights directly
        u, v, _ = svd_cross_product(X, Y)
        n_iter = 1
            
    return u, v, n_iter
    

class _PLS(_PLSBase, RegressorMixin, MultiOutputMixin, metaclass=ABCMeta):
    """Partial Least Squares (PLS)
    
    Base PLS class for regression problems (Mode A, regression and canonical
    variants)
    """
    model = "pls"

    @abstractmethod
    def __init__(self, n_components=2, *, scale=True,
                 deflation_mode="regression", norm_y_weights=False,
                 max_iter=500, tol=1e-06, copy=True):
        super().__init__(n_components=n_components, scale=scale,
                         deflation_mode=deflation_mode, 
                         norm_y_weights=norm_y_weights,
                         max_iter=max_iter, tol=tol, copy=copy)
        
    def weights_estimation(self, X, Y):
        """Estimate PLS weights
        """
        max_iter = self.max_iter
        tol = self.tol
        norm_y_weights = self.norm_y_weights
        
        return _pls_inner_loop(X=X, Y=Y, algorithm="nipals",
                               max_iter=max_iter, tol=tol,
                               norm_y_weights=norm_y_weights)
    
    def _check_sparsity(self):
        """Validates input arguments for sparse extensions of PLS
        """
        
    def _check_blocking(self):
        """Validate blocking inputs for gPLS and sgPLS
        """
        
        
    def fit(self, X, Y):
        """Fit model to data
        """
        super()._fit(X, Y)
        return self
    
    def predict(self, X, copy=True):
        """Fit model to data
        """
        return super()._decision_function(self, X)


class PLSRegression(_PLS):
    """PLS regression
    """

    def __init__(self, n_components=2, *, scale=True,
                 max_iter=500, tol=1e-06, copy=True):
        super().__init__(
            n_components=n_components, scale=scale,
            deflation_mode="regression", norm_y_weights=False,
            max_iter=max_iter, tol=tol, copy=copy)


class PLSCanonical(_PLS):
    """PLS canonical
    """

    def __init__(self, n_components=2, *, scale=True,
                 max_iter=500, tol=1e-06, copy=True):
        super().__init__(
            n_components=n_components, scale=scale,
            deflation_mode="canonical", norm_y_weights=True,
            max_iter=max_iter, tol=tol, copy=copy)
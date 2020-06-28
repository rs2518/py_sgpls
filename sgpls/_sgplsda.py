from abc import abstractmethod

import numpy as np
from ._gplsda import _gPLSDA
from .utils import _check_1d


__all__ = ['sgPLSDACanonical', 'sgPLSDARegression']
   

class _sgPLSDA(_gPLSDA):
    """Sparse Group Partial Least Squares Discriminant Analysis (sgPLS-DA)
    
    Base sparse group PLS class for classification problems (Mode A, 
    regression and canonical variants)
    """
    model = "sgpls"

    @abstractmethod
    def __init__(self, x_block, x_groups, alpha_x,
                 y_block=None, y_groups=None, alpha_y=None,
                 n_components=2, scale=True, method="softmax",
                 deflation_mode="regression",norm_y_weights=False,
                 max_iter=500, tol=1e-06, max_lambda=1e+05,
                 lambda_max_iter=1000, lambda_tol=np.finfo(float).eps**0.25,
                 copy=True):
        super().__init__(x_block, x_groups,
                         y_block=y_block, y_groups=y_groups,
                         n_components=n_components, scale=scale,
                         method=method, deflation_mode=deflation_mode,
                         norm_y_weights=norm_y_weights,
                         max_iter=max_iter, tol=tol, copy=copy)
        self.alpha_x = _check_1d(alpha_x)
        self.alpha_y = alpha_y
        self.max_lambda = max_lambda
        self.lambda_max_iter = lambda_max_iter
        self.lambda_tol = lambda_tol
        
            
class sgPLSDARegression(_sgPLSDA):
    """sgPLSDA Regression
    
    
    """

    def __init__(self, x_block, x_groups, alpha_x,
                 n_components=2, scale=True, method="softmax",
                 max_iter=500, tol=1e-06, max_lambda=1e+05,
                 lambda_max_iter=1000, lambda_tol=np.finfo(float).eps**0.25,
                 copy=True):
        super().__init__(x_block, x_groups, alpha_x,
                         y_block=None, y_groups=None, alpha_y=None,
                         n_components=n_components, scale=scale,
                         method=method, deflation_mode="regression",
                         norm_y_weights=True, max_iter=max_iter, tol=tol,
                         max_lambda=max_lambda,
                         lambda_max_iter=lambda_max_iter,
                         lambda_tol=lambda_tol, copy=copy)


class sgPLSDACanonical(_sgPLSDA):
    """sgPLSDA Canonical
    

    """

    def __init__(self, x_block, x_groups, alpha_x,
                 n_components=2, scale=True, method="softmax",
                 max_iter=500, tol=1e-06, max_lambda=1e+05,
                 lambda_max_iter=1000, lambda_tol=np.finfo(float).eps**0.25,
                 copy=True):
        super().__init__(x_block, x_groups, alpha_x,
                         y_block=None, y_groups=None, alpha_y=None,
                         n_components=n_components, scale=scale,
                         method=method, deflation_mode="canonical",
                         norm_y_weights=True, max_iter=max_iter, tol=tol,
                         max_lambda=max_lambda,
                         lambda_max_iter=lambda_max_iter,
                         lambda_tol=lambda_tol, copy=copy)
import warnings
import numpy as np

from abc import abstractmethod
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array

from ._base import svd_cross_product
from ._pls import _PLS

__all__ = ['sPLSCanonical', 'sPLSRegression']


def _check_1d(arr, warn=True):
    """Check if input is a non-empty 1D array
    """
    arr = check_array(arr, ensure_2d=False)
    
    if arr.ndim == 1:
        return arr
    
    elif arr.ndim == 2 and 1 in arr.shape:
        if warn:
            warnings.warn("Expected a 1-dimensional array",
                          UserWarning)
        
        return arr.ravel()
    
    else:
        raise ValueError("Shape of array is invalid: \n%s\n"
                         "Array must be 1-dimensional"
                         % arr)
        
def _check_1d_or_none(arr, warn=True):
    """Check if input is a non-empty 1D array
    """
    if arr is None:
        pass
    else:
        return _check_1d(arr, warn=warn)
        
def _check_length(arr, max_length, min_length=0):
    """Validate input array length
    """
    if len(arr) < min_length:
        raise ValueError("Invalid array: \n%s\n" 
                         "Length is %d but minimum length is %d"
                         % (arr, len(arr), min_length))
    if len(arr) > max_length:
        raise ValueError("Invalid array: \n%s\n" 
                         "Length is %d but maximum length is %d"
                         % (arr, len(arr), max_length))
        
def _check_entries(arr, max_entry, min_entry=0):
    """Validate input array entries
    """
    if any(arr < min_entry):
        raise ValueError("Invalid array: \n%s\n" 
                         "Entries cannot be below min_entry"
                         "(i.e. Entries must be > %d)"
                         % (arr, min_entry))
    if any(arr > max_entry):
        raise ValueError("Invalid array: \n%s\n" 
                         "Entries cannot exceed max_entry"
                         "(i.e. Entries must be < %d)"
                         % (arr, max_entry))
        
def _soft_thresholding(y, lambda_k):
    """Soft-thresholding function
    """
    eps = np.finfo(float).eps ** 0.5
    a = np.abs(y) - lambda_k
    a[a < eps] = 0
    return np.sign(y) * a


def _spls_inner_loop(X, Y, nx_var, ny_var, max_iter=500, tol=1e-06,
                     norm_y_weights=True):
    """Inner loop for sPLS weights estimation
    """
    u_old, v_old, M = svd_cross_product(X, Y)
    eps = np.finfo(X.dtype).eps
    
    x_sparsity = X.shape[1] - nx_var
    y_sparsity = Y.shape[1] - ny_var
    
    for i in range(max_iter):
        # 1.1 Calculate M_v : the X projections
        M_v = np.dot(M, v_old)
        # 1.2 Find lambda_x : the X penalty
        if x_sparsity == 0:
            lambda_x = 0
        else:
            lambda_x = sorted(np.absolute(M_v))[x_sparsity-1]
        # The number of non-zero X loadings gives the appropriate value
        # for penalisation of X variables.
        # 1.3 Update u : the X weights
        u = _soft_thresholding(M_v, lambda_x)
        if np.dot(u.T, u) < eps:
            u += eps
        # 1.4 Normalise u
        u /= np.sqrt(np.dot(u.T, u)) + eps
        
        # 2.1 Calculate M_u : the Y projections
        M_u = np.dot(M.T, u)
        # 2.2 Find lambda_y : the Y penalty
        if y_sparsity == 0:
            lambda_y = 0
        else:
            lambda_y = sorted(np.absolute(M_u))[y_sparsity-1]
        # 2.3 Update v : the Y weights
        v = _soft_thresholding(M_u, lambda_y)
        # 2.4 Normalise v
        if norm_y_weights:
            v /= np.sqrt(np.dot(v.T, v)) + eps
        
        u_diff = u - u_old
        v_diff = v - v_old
        if np.dot(u_diff.T, u_diff) < tol and np.dot(v_diff.T, v_diff) < tol:
            break
        u_old = u
        v_old = v
    
    n_iter = i + 1
    if n_iter == max_iter:
        warnings.warn("Maximum number of iterations reached",
                      ConvergenceWarning)
        
    return u, v, n_iter
    

class _sPLS(_PLS):
    """Sparse Partial Least Squares (sPLS)
    """
    model = "spls"

    @abstractmethod
    def __init__(self, x_vars, y_vars=None, n_components=2, *, scale=True,
                 deflation_mode="regression", norm_y_weights=False,
                 max_iter=500, tol=1e-06, copy=True):
        super().__init__(n_components=n_components, scale=scale,
                         deflation_mode=deflation_mode, 
                         norm_y_weights=norm_y_weights,
                         max_iter=max_iter, tol=tol, copy=copy)
        if x_vars is None and y_vars is None:
            raise ValueError("Must include a sparsity parameter. "
                             "Use PLS instead for zero sparsity")
        self.x_vars = _check_1d_or_none(x_vars)
        self.y_vars = _check_1d_or_none(y_vars)

    def weights_estimation(self, X, Y, **kwargs):
        """Estimate PLS weights
        """
        k = self._comp
        
        nx_var = self._x_vars[k]
        ny_var = self._y_vars[k]
        max_iter = self.max_iter
        tol = self.tol
        norm_y_weights = self.norm_y_weights
        
        return _spls_inner_loop(X=X, Y=Y, nx_var=nx_var, ny_var=ny_var,
                                max_iter=max_iter, tol=tol,
                                norm_y_weights=norm_y_weights)
            
    def _check_sparsity(self, X, Y):
        """Validates input arguments for sparse extensions of PLS
        """
        # Validate arrays
        n = self.n_components
        p = X.shape[1]
        if Y.ndim == 1:
            q = 1
        else:
            q = Y.shape[1]
        
        _check_length(self.x_vars, max_length=n, min_length=n)        
        _check_entries(self.x_vars, max_entry=p, min_entry=1)
        
        _check_length(self.y_vars, max_length=n, min_length=n)
        _check_entries(self.y_vars, max_entry=q, min_entry=1)
        
        # Assign sparsity parameters to preserve None
        if self.x_vars is None:
            self._x_vars = np.full(n, p)
        else:
            self._x_vars = self.x_vars

        if self.y_vars is None:
            self._y_vars = np.full(n, q)
        else:
            self._y_vars = self.y_vars
        
        return self
        
        
    def fit(self, X, Y):
        """Fit model to data
        """
        self._check_sparsity(X, Y)
        
        super().fit(X, Y)
        return self        
        

class sPLSRegression(_sPLS):
    """sPLS regression
    """

    def __init__(self, x_vars, y_vars=None, n_components=2, *,
                 scale=True, max_iter=500, tol=1e-06, copy=True):
        super().__init__(x_vars, y_vars=y_vars,
                         n_components=n_components, scale=scale,
                         deflation_mode="regression",
                         norm_y_weights=False,
                         max_iter=max_iter, tol=tol, copy=copy)


class sPLSCanonical(_sPLS):
    """sPLS canonical
    """

    def __init__(self, x_vars, y_vars=None, n_components=2, *, scale=True,
                 max_iter=500, tol=1e-06, copy=True):
        super().__init__(x_vars, y_vars=y_vars,
                         n_components=n_components, scale=scale,
                         deflation_mode="canonical",
                         norm_y_weights=True,
                         max_iter=max_iter, tol=tol, copy=copy)
from abc import abstractmethod

from ._plsda import _PLSDA
from .utils import _check_1d


__all__ = ['sPLSDACanonical', 'sPLSDARegression']


class _sPLSDA(_PLSDA):
    """Sparse Partial Least Squares Discriminant Analysis (sPLS-DA)
    
    Base sparse PLS class for classification problems (Mode A, regression and
    canonical variants)
    """
    model = "spls"
    
    @abstractmethod
    def __init__(self, x_vars, y_vars=None, n_components=2, scale=True,
                 deflation_mode="regression", norm_y_weights=False,
                 max_iter=500, tol=1e-06, copy=True):
        super().__init__(n_components=n_components, scale=scale,
                         deflation_mode=deflation_mode, algorithm="NA",
                         norm_y_weights=norm_y_weights,
                         max_iter=max_iter, tol=tol, copy=copy)
        self.x_vars = _check_1d(x_vars)
        self.y_vars = y_vars


class sPLSDARegression(_sPLSDA):
    """sPLSDA Regression
      

    """

    def __init__(self, x_vars, n_components=2, scale=True, max_iter=500,
                 tol=1e-06, copy=True):
        super().__init__(
            x_vars, y_vars=None, n_components=n_components,
            scale=scale, deflation_mode="regression",
            norm_y_weights=True, max_iter=max_iter,
            tol=tol, copy=copy)


class sPLSDACanonical(_sPLSDA):
    """sPLSDA Canonical


    """

    def __init__(self, x_vars, n_components=2, scale=True, max_iter=500,
                 tol=1e-06, copy=True):
        super().__init__(
            x_vars, y_vars=None, n_components=n_components,
            scale=scale, deflation_mode="canonical",
            norm_y_weights=True, max_iter=max_iter,
            tol=tol, copy=copy)
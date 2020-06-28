from abc import abstractmethod

from ._plsda import _PLSDA
from .utils import _check_1d


__all__ = ['gPLSDACanonical', 'gPLSDARegression']
   

class _gPLSDA(_PLSDA):
    """Group Partial Least Squares Discriminant Analysis (gPLS-DA)
        
    Base group PLS class for classification problems (Mode A, regression and
    canonical variants)
    """
    model = "gpls"

    @abstractmethod
    def __init__(self, x_block, x_groups, y_block=None, y_groups=None,
                 n_components=2, scale=True, method="softmax",
                 deflation_mode="regression", norm_y_weights=False,
                 max_iter=500, tol=1e-06, copy=True):
        super().__init__(n_components=n_components, scale=scale,
                         method=method, deflation_mode=deflation_mode,
                         algorithm="NA", norm_y_weights=norm_y_weights,
                         max_iter=max_iter, tol=tol, copy=copy)        
        self.x_block = _check_1d(x_block)
        self.x_groups = _check_1d(x_groups)
        self.y_block = y_block
        self.y_groups = y_groups
        
        
class gPLSDARegression(_gPLSDA):
    """gPLSDA Regression
    

    """

    def __init__(self, x_block, x_groups, n_components=2, scale=True,
                 method="softmax", max_iter=500, tol=1e-06, copy=True):
        super().__init__(
            x_block, x_groups, y_block=None, y_groups=None,
            n_components=n_components, scale=scale, method=method,
            deflation_mode="regression", norm_y_weights=True,
            max_iter=max_iter, tol=tol, copy=copy)


class gPLSDACanonical(_gPLSDA):
    """gPLSDA Canonical


    """

    def __init__(self, x_block, x_groups, n_components=2, scale=True,
                 method="softmax", max_iter=500, tol=1e-06, copy=True):
        super().__init__(
            x_block, x_groups, y_block=None, y_groups=None,
            n_components=n_components, scale=scale, method=method,
            deflation_mode="canonical", norm_y_weights=True,
            max_iter=max_iter, tol=tol, copy=copy)
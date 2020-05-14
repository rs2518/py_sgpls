from abc import abstractmethod

from sklearn.base import ClassifierMixin

from ._pls import _PLSBase
# from .utils import _check_1d

__all__ = ['PLSDACanonical', 'PLSDARegression']


class _PLSDA(_PLSBase, ClassifierMixin):
    """Partial Least Squares Discriminant Analysis (PLS-DA)
    
    Base PLS class for classification problems (Mode A, regression and
    canonical variants)
    """
    model = "pls"
    
    @abstractmethod
    def __init__(self, n_components=2, scale=True, 
                 deflation_mode="regression", algorithm="nipals",
                 norm_y_weights=False, max_iter=500, tol=1e-06, copy=True):
        super().__init__(n_components=n_components, scale=scale,
                         deflation_mode=deflation_mode, algorithm=algorithm,
                         norm_y_weights=norm_y_weights,
                         max_iter=max_iter, tol=tol, copy=copy)
        
    def fit(self, X, Y):
        """Fit model to data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.
            
        # Y : array-like of shape (n_samples, n_targets)
        #     Target vectors, where n_samples is the number of samples and
        #     n_targets is the number of response variables.
        """
        ## Preprocess Y matrix to support classification
        super().fit(X, Y)
        self.model += 'da'

        return self


class PLSDARegression(_PLSDA):
    """PLSDA Regression
    
    PLS-DA with 'regression' deflation_mode
    """

    def __init__(self, n_components=2, scale=True, algorithm="nipals",
                 max_iter=500, tol=1e-06, copy=True):
        super().__init__(
            n_components=n_components, scale=scale,
            deflation_mode="regression", algorithm=algorithm,
            norm_y_weights=False, max_iter=max_iter, tol=tol,
            copy=copy)


class PLSDACanonical(_PLSDA):
    """ PLSDA Canonical
    
    PLS-DA with 'canonical' deflation_mode
    """

    def __init__(self, n_components=2, scale=True, algorithm="nipals",
                 max_iter=500, tol=1e-06, copy=True):
        super().__init__(
            n_components=n_components, scale=scale,
            deflation_mode="canonical", algorithm=algorithm,
            norm_y_weights=True, max_iter=max_iter, tol=tol,
            copy=copy)
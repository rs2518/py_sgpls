from abc import abstractmethod

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

from ._pls import _PLSBase
from .utils import _check_1d

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
        # Preprocess Y to support classification
        # NOTE: Algorithm is simplified to implementing PLS1DA for binary case
        # and PLS2DA for multi-class case.
        # In reality, one may choose to PLS1DA for a multi-class case
        # - equivalent to a 'one-vs-rest' (OVR) scheme. See sklearn's
        # logistic regression 'multiclass' implementation
        Y = _check_1d(Y)
        check_classification_targets(Y)
        
        target_type = type_of_target(Y)
        if target_type not in ('binary', 'multiclass'):
            raise ValueError("Unsupported classification target type")
                
        self.classes_ = np.unique(Y)
        if self.classes_.size < 2:
            raise ValueError("Y must contain data from at least 2 classes")
        if self.classes_.size == 2:
            le = LabelEncoder()
            Ys = le.fit_transform(Y)
        else:
            lb = LabelBinarizer()
            Ys = lb.fit_transform(Y)
        
        super().fit(X, Ys)
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
            norm_y_weights=True, max_iter=max_iter, tol=tol,
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
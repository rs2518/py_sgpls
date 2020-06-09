from abc import abstractmethod

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics.pairwise import pairwise_distances_argmin
from scipy.linalg import pinv2

from ._pls import _PLSBase
from .utils import _check_1d

__all__ = ['PLSDACanonical', 'PLSDARegression']


class _PLSDA(_PLSBase, ClassifierMixin):
    """Partial Least Squares Discriminant Analysis (PLS-DA)
    
    Base PLS class for classification problems (Mode A, regression and
    canonical variants)
    
    Note that scaling often leads to slight differences in the accuracy of the
    predictions given from the fitted model.
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
        # NOTE: Algorithm is simplified to restrict to single label outputs, 
        # implementing PLS1DA for binary (two classes) case and PLS2DA for
        # multi-class case.
        # In reality, one may choose PLS1DA for a multi-class case
        # - equivalent to a 'one-vs-rest' (OVR) scheme.
        Y = _check_1d(Y)
        check_classification_targets(Y)
        target_type = type_of_target(Y)
        if target_type not in ('binary', 'multiclass'):
            raise ValueError("Unsupported classification target type")
        lb = LabelBinarizer()
        Ys = lb.fit_transform(Y)
        # if Ys.shape[1] == 1:
        #     Ys = np.hstack([1 - Ys, Ys])   # Convert binary case to PLSDA2
        
        self.classes_ = lb.classes_
        if len(self.classes_) < 2:
            raise ValueError("Y must contain data from at least 2 classes")
        super().fit(X, Ys)
        
        # Calculate centroids matrix (mean coordinate of the fitted latent
        # component projection for each class) and matrix of centred
        # projections (fitted latent component projection minus the active
        # classes centroid)
        T = np.dot(X, self.x_rotations_)
        self._centroids = np.dot(Ys.T, T) / Ys.sum(axis=0).reshape(-1,1)
        self._centred_projection = T - np.dot(Ys, self._centroids)
        
        self.model += 'da'
        
        return self
    
    def predict(self, X, metric="max", copy=True):
        """Apply the dimension reduction learned on the train data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.
            
        metric : string, "max", "euclidean" or "mahalanobis", (default="max")
            Metric used to predict labels.
            
            "max" : selects the class corresponding to the dummy variable
            with the greatest predicted value
            
            "euclidean" : selects the class belonging to the centroid with
            the shortest euclidean distance to the predicted latent projection
            
            "mahalanobis" : selects the class belonging to the centroid with
            the shortest mahalanobis distance to the predicted latent
            projection
            
        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.
        """
        if metric not in ("max", "euclidean", "mahalanobis"):
            raise ValueError("Incompatible configuration: '%s' metric is "
                             "unsupported or invalid" % metric)
        
        if metric == "max":
            Ypred = super().predict(X, copy=copy).argmax(axis=1)
        else:
            check_is_fitted(self)
            Tpred = np.dot(X, self.x_rotations_)
            kwargs = {}
            
            if metric == "mahalanobis":
                V = np.cov(self._centred_projection)
                inv_V = pinv2(V)
                kwargs["VI"] = inv_V
            
            Ypred = pairwise_distances_argmin(Tpred, self._centroids,
                                              axis=1, metric=metric,
                                              metric_kwargs=kwargs)
        
        return Ypred
    
    
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
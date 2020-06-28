from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.utils.extmath import softmax
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.naive_bayes import GaussianNB
from scipy.linalg import pinv2

from ._base import _PLSBase
from .utils import _check_1d

__all__ = ['PLSDACanonical', 'PLSDARegression']


class _PLSDA(_PLSBase, ClassifierMixin, metaclass=ABCMeta):
    """Partial Least Squares Discriminant Analysis (PLS-DA)
    
    Base PLS class for classification problems (Mode A, regression and
    canonical variants)
    
    Note that scaling often leads to slight differences in the accuracy of the
    predictions given from the fitted model.
    
    ####
    method : string, "softmax", "naive_bayes", "euclidean" or mahalanobis",
    (default="softmax")
        
        Metric used to predict labels.
        
        "softmax" : selects the class corresponding to the dummy variable
        with the greatest probability. The 'probability' is assigned from
        transforming the model predictions with the softmax function.
        
        "naive_bayes" : selects the class corresponding to the dummy
        variable with the greatest probability. The 'probability' is
        assigned from the posterior derived from running a Naive Bayes
        classifier on the model predictions. Posterior probabilities are
        assumed to follow a Gaussian distribution.
        
        "euclidean" : selects the class belonging to the centroid with
        the shortest euclidean distance to the predicted latent projection
        
        "mahalanobis" : selects the class belonging to the centroid with
        the shortest mahalanobis distance to the predicted latent
        projection
    
    """
    model = "pls"
    
    @abstractmethod
    def __init__(self, n_components=2, scale=True,
                 deflation_mode="regression", algorithm="nipals",
                 method="softmax", norm_y_weights=False,
                 max_iter=500, tol=1e-06, copy=True):
        super().__init__(n_components=n_components, scale=scale,
                         deflation_mode=deflation_mode, algorithm=algorithm,
                         norm_y_weights=norm_y_weights,
                         max_iter=max_iter, tol=tol, copy=copy)
        self.method = method
        
    def _binarise_target(self, Y, return_classes=False):
        """Convert target vector into matrix of binary dummy variables
        """
        lb = LabelBinarizer()
        Ys = lb.fit_transform(Y)
        # if Ys.shape[1] == 1:
        #     Ys = np.hstack([1 - Ys, Ys])   # Convert binary case to PLSDA2
        
        if return_classes:
            self.classes_ = lb.classes_
            
        return Ys
            
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
        
        if self.method not in ("softmax", "naive_bayes", "euclidean",
                          "mahalanobis"):
            raise ValueError("Incompatible configuration: '%s' method is "
                             "unsupported or invalid" % self.method)
        self._x = X.copy()
        self._y = Y.copy()

        Ys = self._binarise_target(Y, return_classes=True)
        if len(self.classes_) < 2:
            raise ValueError("Y must contain data from at least 2 classes")
        super()._fit(X, Ys)
        
        
        self.model += 'da'
        
        return self
        
    def _predict_dist(self, X):
        """Predict labels using distance-based metrics.
        """
        check_is_fitted(self)
        
        # Calculate centroids matrix and latent projections
        Ys = self._binarise_target(self._y)
        T = self.transform(self._x, copy=False)
        centroids = np.dot(Ys.T, T) / Ys.sum(axis=0).reshape(-1,1)
        Tpred = self.transform(X)
        
        if self.method == "euclidean":
            kwargs = {}

        elif self.method == "mahalanobis":
            V = np.cov((T - np.dot(Ys, centroids)).T)
            inv_V = pinv2(V)
            kwargs = {"VI": inv_V}
            
        labels, self.min_distances = pairwise_distances_argmin_min(
            Tpred, centroids, axis=1, metric=self.method, metric_kwargs=kwargs)
        return labels

    def _predict_proba(self, X):
        """Probability estimation for PLSDA.
        
        Converts the raw PLS prediction results into 
        # Use softmax or Naive Bayes to predict labels via probablistic methods.
        # Softmax is akin to max.dist method
        """
        check_is_fitted(self) 
        
        decision = self._decision_function(X)
        if self.method == "softmax":
            if decision.ndim == 1:
                # Workaround for binary outcomes which requires prediction
                # with only a 1D decision.
                decision_2d = np.c_[-decision, decision]
            else:
                decision_2d = decision
            prob = softmax(decision_2d)
        
        elif self.method == "naive_bayes":
            # # Ideally Bayes Classifier with Kernel Density Estimation should
            # # be used because estimation is non-parametric (no heavy
            # # assumptions on the distribution of conditional probabilities).
            # # Use GaussianNB for now!
            
            # Train Bayes classifier (learn posterior probabilities on
            # training data) then predict on testing data X          
            gnb = GaussianNB()
            gnb.fit(self._decision_function(self._x), self._y)
            
            prob = gnb.predict_proba(decision)

        return prob
    
    def predict(self, X):
        """
        Predict class for samples in X using the selected method.
        
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        C : array, shape [n_samples]
            Predicted class label per sample.
        """
        if self.method in ("euclidean", "mahalanobis"):
            Y_pred = self._predict_dist(X)
            
        elif self.method in ("softmax", "naive_bayes"):
            prob = self._predict_proba(X)
            Y_pred = prob.argmax(axis=1)
        return Y_pred
    
    
class PLSDARegression(_PLSDA):
    """PLSDA Regression
    
    PLS-DA with 'regression' deflation_mode
    """

    def __init__(self, n_components=2, scale=True, algorithm="nipals",
                 method="softmax", max_iter=500, tol=1e-06, copy=True):
        super().__init__(
            n_components=n_components, scale=scale,
            deflation_mode="regression", algorithm=algorithm,
            method=method, norm_y_weights=True,
            max_iter=max_iter, tol=tol, copy=copy)


class PLSDACanonical(_PLSDA):
    """ PLSDA Canonical
    
    PLS-DA with 'canonical' deflation_mode
    """

    def __init__(self, n_components=2, scale=True, algorithm="nipals",
                 method="softmax", max_iter=500, tol=1e-06, copy=True):
        super().__init__(
            n_components=n_components, scale=scale,
            deflation_mode="canonical", algorithm=algorithm,
            method=method, norm_y_weights=True,
            max_iter=max_iter, tol=tol, copy=copy)
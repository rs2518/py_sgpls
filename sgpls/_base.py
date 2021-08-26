import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.linalg import pinv2, svd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.cross_decomposition._pls import _center_scale_xy, _svd_flip_1d


def svd_cross_product(X, Y):
    """Returns singular vectors and matrix product from X'Y
    """
    C = np.dot(X.T, Y)
    U, s, Vt = svd(C, full_matrices=False)
    return U[:, 0], Vt[0, :], C


class _PLSBase(TransformerMixin, BaseEstimator, metaclass=ABCMeta):
    """Partial Least Squares (PLS)
    """
    
    @abstractmethod
    def __init__(self, n_components=2, *, scale=True,
                 deflation_mode="regression", norm_y_weights=False,
                 max_iter=500, tol=1e-06, copy=True):
        self.n_components = n_components
        self.deflation_mode = deflation_mode
        self.norm_y_weights = norm_y_weights
        self.scale = scale
        # self.algorithm = algorithm    # Fix algorithm to SVD
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy
    
    @abstractmethod
    def weights_estimation(self, X, Y):
        """Calculates optimal weights
        """
    
    @abstractmethod
    def _check_sparsity(self):
        """Validates input arguments for sparse extensions of PLS
        """
        
    @abstractmethod
    def _check_blocking(self):
        """Validate blocking inputs for gPLS and sgPLS
        """
        
        
    def _fit(self, X, Y):
        """Fit model to data
        """

        # copy since this will contains the residuals (deflated) matrices
        check_consistent_length(X, Y)
        X = check_array(X, dtype=np.float64, copy=self.copy,
                        ensure_min_samples=2)
        Y = check_array(Y, dtype=np.float64, copy=self.copy, ensure_2d=False)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n = X.shape[0]
        p = X.shape[1]
        q = Y.shape[1]

        # Input validation
        # ----------------
        n_components = self.n_components
        if self.deflation_mode == 'regression':
            # With PLSRegression n_components is bounded by the rank of (X.T X)
            # see Wegelin page 25
            rank_upper_bound = p
            if not 1 <= n_components <= rank_upper_bound:
                # TODO: raise an error in 1.1
                warnings.warn(
                    f"As of version 0.24, n_components({n_components}) should "
                    f"be in [1, n_features]."
                    f"n_components={rank_upper_bound} will be used instead. "
                    f"In version 1.1 (renaming of 0.26), an error will be "
                    f"raised.",
                    FutureWarning
                )
                n_components = rank_upper_bound
        else:
            # With CCA and PLSCanonical, n_components is bounded by the rank of
            # X and the rank of Y: see Wegelin page 12
            rank_upper_bound = min(n, p, q)
            if not 1 <= self.n_components <= rank_upper_bound:
                # TODO: raise an error in 1.1
                warnings.warn(
                    f"As of version 0.24, n_components({n_components}) should "
                    f"be in [1, min(n_features, n_samples, n_targets)] = "
                    f"[1, {rank_upper_bound}]. "
                    f"n_components={rank_upper_bound} will be used instead. "
                    f"In version 1.1 (renaming of 0.26), an error will be "
                    f"raised.",
                    FutureWarning
                )
                n_components = rank_upper_bound

            
        self._norm_y_weights = (self.deflation_mode == 'canonical')  # 1.1
        norm_y_weights = self._norm_y_weights            
                            
        # Scale (in place)
        Xk, Yk, self.x_mean_, self.y_mean_, self._x_std, self._y_std = (
            _center_scale_xy(X, Y, self.scale))
        
        # Results matrices
        self.x_scores_ = np.zeros((n, self.n_components))
        self.y_scores_ = np.zeros((n, self.n_components))
        self.x_weights_ = np.zeros((p, self.n_components))
        self.y_weights_ = np.zeros((q, self.n_components))
        self.x_loadings_ = np.zeros((p, self.n_components))
        self.y_loadings_ = np.zeros((q, self.n_components))
        self.n_iter_ = []
        
        # Outer loop, over components
        for k in range(self.n_components):

            # 1) weights estimation (inner loop)
            # -----------------------------------
            try:
                x_weights, y_weights, n_iter_ = \
                    self.weights_estimation(X, Y)
            except StopIteration as e:
                if str(e) != "Y residual is constant":
                    raise
                warnings.warn(f"Y residual is constant at iteration {k}")
                break
                
            self.n_iter_.append(n_iter_)
                    
            # inplace sign flip for consistency across solvers and archs
            _svd_flip_1d(x_weights, y_weights)

            # compute scores, i.e. the projections of X and Y
            x_scores = np.dot(Xk, x_weights)
            if norm_y_weights:
                y_ss = 1
            else:
                y_ss = np.dot(y_weights, y_weights)
            y_scores = np.dot(Yk, y_weights) / y_ss
            
            
            # 2) Deflation (in place)
            # ----------------------
            x_loadings = np.dot(x_scores, Xk) / np.dot(x_scores, x_scores)
            Xk -= np.outer(x_scores, x_loadings)

            if self.deflation_mode == "canonical":
                # regress Yk on y_score
                y_loadings = np.dot(y_scores, Yk) / np.dot(y_scores, y_scores)
                Yk -= np.outer(y_scores, y_loadings)
            if self.deflation_mode == "regression":
                # regress Yk on x_score
                y_loadings = np.dot(x_scores, Yk) / np.dot(x_scores, x_scores)
                Yk -= np.outer(x_scores, y_loadings)

            self.x_weights_[:, k] = x_weights
            self.y_weights_[:, k] = y_weights
            self.x_scores_[:, k] = x_scores
            self.y_scores_[:, k] = y_scores
            self.x_loadings_[:, k] = x_loadings
            self.y_loadings_[:, k] = y_loadings

        # X was approximated as Xi . Gamma.T + X_(R+1)
        # Xi . Gamma.T is a sum of n_components rank-1 matrices. X_(R+1) is
        # whatever is left to fully reconstruct X, and can be 0 if X is of rank
        # n_components.
        # Similiarly, Y was approximated as Omega . Delta.T + Y_(R+1)

        # Compute transformation matrices (rotations_). See User Guide.
        self.x_rotations_ = np.dot(
            self.x_weights_,
            pinv2(np.dot(self.x_loadings_.T, self.x_weights_),
                  check_finite=False))
        self.y_rotations_ = np.dot(
            self.y_weights_, pinv2(np.dot(self.y_loadings_.T, self.y_weights_),
                                   check_finite=False))

        self.coef_ = np.dot(self.x_rotations_, self.y_loadings_.T)
        self.coef_ = self.coef_ * self._y_std
        
        return self

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction learned on the train data
        """
        check_is_fitted(self)
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        # Normalize
        X -= self.x_mean_
        X /= self.x_std_
        # Apply rotation
        x_scores = np.dot(X, self.x_rotations_)
        if Y is not None:
            Y = check_array(Y, ensure_2d=False, copy=copy, dtype=FLOAT_DTYPES)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            Y -= self.y_mean_
            Y /= self.y_std_
            y_scores = np.dot(Y, self.y_rotations_)
            return x_scores, y_scores

        return x_scores

    def inverse_transform(self, X):
        """Transform data back to its original space
        """
        check_is_fitted(self)
        X = check_array(X, dtype=FLOAT_DTYPES)
        # From pls space to original space
        X_reconstructed = np.matmul(X, self.x_loadings_.T)

        # Denormalize
        X_reconstructed *= self.x_std_
        X_reconstructed += self.x_mean_
        return X_reconstructed

    def _decision_function(self, X, copy=True):
        """Apply the dimension reduction learned on the train data
        """
        check_is_fitted(self)
        X = self._validate_data(X, copy=copy, dtype=FLOAT_DTYPES, reset=False)
        # Normalize
        X -= self.x_mean_
        X /= self.x_std_
        Ypred = np.dot(X, self.coef_)
        return Ypred + self.y_mean_

    def fit_transform(self, X, y=None):
        """Learn and apply the dimension reduction on the train data
        """
        return self.fit(X, y).transform(X, y)

    def _more_tags(self):
        return {'poor_score': True}

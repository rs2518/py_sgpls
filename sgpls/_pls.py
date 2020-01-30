import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.linalg import pinv2
from scipy.optimize import brentq

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.base import MultiOutputMixin
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.extmath import svd_flip
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.exceptions import ConvergenceWarning
from sklearn.cross_decomposition.pls_ import _center_scale_xy

from .utils import svd_cross_product, sparsity_conversion
from .utils import _soft_thresholding, _group_thresholding
from .utils import _lambda_quadratic, _sparse_group_thresholding
from .utils import pls_blocks, pls_array

__all__ = ['PLSCanonical', 'PLSRegression']


def _nipals_twoblocks_inner_loop(X, Y, mode="A", max_iter=500, tol=1e-06,
                                 norm_y_weights=False):
    """Inner loop of the iterative NIPALS algorithm.
    
    Provides an alternative to the svd(X'Y); returns the first left and right
    singular vectors of X'Y.  See PLS for the meaning of the parameters.  It is
    similar to the Power method for determining the eigenvectors and
    eigenvalues of a X'Y.
    """
    for col in Y.T:
        if np.any(np.abs(col) > np.finfo(np.double).eps):
            y_score = col.reshape(len(col), 1)
            break

    x_weights_old = 0
    ite = 1
    X_pinv = Y_pinv = None
    eps = np.finfo(X.dtype).eps
    # Inner loop of the Wold algo.
    while True:
        # 1.1 Update u: the X weights
        if mode == "B":
            if X_pinv is None:
                # We use slower pinv2 (same as np.linalg.pinv) for stability
                # reasons
                X_pinv = pinv2(X, check_finite=False)
            x_weights = np.dot(X_pinv, y_score)
        else:  # mode A
            # Mode A regress each X column on y_score
            x_weights = np.dot(X.T, y_score) / np.dot(y_score.T, y_score)
        # If y_score only has zeros x_weights will only have zeros. In
        # this case add an epsilon to converge to a more acceptable
        # solution
        if np.dot(x_weights.T, x_weights) < eps:
            x_weights += eps
        # 1.2 Normalize u
        x_weights /= np.sqrt(np.dot(x_weights.T, x_weights)) + eps
        # 1.3 Update x_score: the X latent scores
        x_score = np.dot(X, x_weights)
        # 2.1 Update y_weights
        if mode == "B":
            if Y_pinv is None:
                Y_pinv = pinv2(Y, check_finite=False)  # compute once pinv(Y)
            y_weights = np.dot(Y_pinv, x_score)
        else:
            # Mode A regress each Y column on x_score
            y_weights = np.dot(Y.T, x_score) / np.dot(x_score.T, x_score)
        # 2.2 Normalize y_weights
        if norm_y_weights:
            y_weights /= np.sqrt(np.dot(y_weights.T, y_weights)) + eps
        # 2.3 Update y_score: the Y latent scores
        y_score = np.dot(Y, y_weights) / (np.dot(y_weights.T, y_weights) + eps)
        # y_score = np.dot(Y, y_weights) / np.dot(y_score.T, y_score) ## BUG
        x_weights_diff = x_weights - x_weights_old
        if np.dot(x_weights_diff.T, x_weights_diff) < tol or Y.shape[1] == 1:
            break
        if ite == max_iter:
            warnings.warn('Maximum number of iterations reached',
                          ConvergenceWarning)
            break
        x_weights_old = x_weights
        ite += 1
    return x_weights, y_weights, ite


def _spls_inner_loop(X, Y, x_var, y_var, max_iter=500, tol=1e-06,
                    norm_y_weights=True):
    """Inner loop for iterative tuning of sPLS algorithm.
    
    Estimates PLS weights which solve the sparse PLS objective function.
    See Lê Cao et al (2008) for details.
    """
    u_old, v_old, M = svd_cross_product(X, Y, return_matrix=True)
    ite = 1
    eps = np.finfo(X.dtype).eps
    # Inner loop of sPLS
    while True:
        # 1.1 Calculate M_v : the X projections
        M_v = np.dot(M, v_old)
        # 1.2 Find lambda_x : the X penalty
        if x_var == 0:
            lambda_x = 0
        else:
            lambda_x = sorted(np.absolute(M_v))[x_var]
        # The number of non-zero X loadings gives the appropriate value
        # for penalisation of X variables.
        # 1.3 Update u : the X weights
        u = _soft_thresholding(M_v, lambda_x)
        # 1.4 Normalise u
        u /= np.sqrt(np.dot(u.T, u)) + eps
        
        # 2.1 Calculate M_u : the Y projections
        M_u = np.dot(M.T, u)
        # 2.2 Find lambda_y : the Y penalty
        if y_var == 0:
            lambda_y = 0
        else:
            lambda_y = sorted(np.absolute(M_u))[y_var]
        # 2.3 Update v : the Y weights
        v = _soft_thresholding(M_u, lambda_y)
        # 2.4 Normalise v
        if norm_y_weights:
            v /= np.sqrt(np.dot(v.T, v)) + eps
        
        u_diff = u - u_old
        v_diff = v - v_old
        if np.dot(u_diff.T, u_diff) < tol and np.dot(v_diff.T, v_diff) < tol:
            break
        if ite == max_iter:
            warnings.warn('Maximum number of iterations reached',
                          ConvergenceWarning)
            break
        u_old = u
        v_old = v
        ite += 1
    return u, v, ite


def _gpls_inner_loop(X, Y, x_group, y_group, x_ind, y_ind,
                       max_iter=500, tol=1e-06, norm_y_weights=True):
    """Inner loop for iterative tuning of gPLS algorithm.
    
    Estimates PLS weights which solve the group PLS objective function.
    See Benoît Liquet et al (2015) for details.
    """                
    u_old, v_old, M = svd_cross_product(X, Y, return_matrix=True)
    ite = 1
    eps = np.finfo(X.dtype).eps
    
    u = np.zeros_like(u_old)
    v = np.zeros_like(v_old)
    k = len(x_ind) + 1
    l = len(y_ind) + 1
    x_penalty = np.zeros(k)
    y_penalty = np.zeros(l)
    x_range = [range(x_ind[i], x_ind[i+1]) for i in range(k)]
    y_range = [range(y_ind[i], y_ind[i+1]) for i in range(l)]
    # Inner loop of gPLS
    while True:      
        # 1.1 Calculate M_v : the X projections
        M_v = np.dot(M, v_old)
        # 1.2 Calculate contribution of X groups to M_v
        for group in range(k):
            arr = np.array(M_v[x_range[group]])
            x_penalty[group] = 2 * np.sqrt(np.dot(arr.T, arr))
            x_penalty[group] /= np.sqrt(len(arr))
        # 1.3 Find lambda_x : the X penalty
        if x_group == 0:
            lambda_x = 0
        else:
            lambda_x = sorted(np.absolute(x_penalty))[x_group]        
        # Groups of X variables are penalised using the group thresholding
        # function and the appropriate penalty calculated from the magnitude
        # of the projections for each group.
        # 1.4 Update u : the X weights        
        for group in range(k):
            arr = np.array(M_v[x_range[group]])
            u[x_range[group]] = _group_thresholding(
                    arr, lambda_x, x_penalty[group])
        # 1.5 Normalise u
        u /= np.sqrt(np.dot(u.T, u)) + eps
        
        # 2.1 Calculate M_u : the Y projections
        M_u = np.dot(M.T, u)
        # 2.2 Calculate contribution of Y groups to M_u
        for group in range(l):
            arr = np.array(M_u[y_range[group]])
            y_penalty[group] = 2 * np.sqrt(np.dot(arr.T, arr))
            y_penalty[group] /= np.sqrt(len(arr))
        # 2.3 Find lambda_y : the Y penalty
        if y_group == 0:
            lambda_y = 0
        else:
            lambda_y = sorted(np.absolute(y_penalty))[y_group]        
        # 2.4 Update v : the Y weights
        for group in range(l):
            arr = np.array(M_u[y_range[group]])
            v[y_range[group]] = _group_thresholding(
                    arr, lambda_y, y_penalty[group])
        # 2.5 Normalise u
        if norm_y_weights:
            v /= np.sqrt(np.dot(v.T, v)) + eps        
        
        u_diff = u - u_old
        v_diff = v - v_old
        if np.dot(u_diff.T, u_diff) < tol and np.dot(v_diff.T, v_diff) < tol:
            break
        if ite == max_iter:
            warnings.warn('Maximum number of iterations reached',
                          ConvergenceWarning)
            break
        u_old = u
        v_old = v
        ite += 1
    return u, v, ite


def _sgpls_inner_loop(X, Y, x_group, y_group, x_ind, y_ind,
                     alpha_x, alpha_y, max_iter=500,
                     tol=1e-06, norm_y_weights=True,
                     lambda_tol=np.finfo(float).eps**0.25,
                     max_lambda=1e+05, lambda_niter=1000):
    """Inner loop for iterative tuning of sgPLS algorithm.
    
    Estimates PLS weights which solve the sparse group PLS objective function.
    See Benoît Liquet et al (2015) for details.  
    Lambda thresholds are solved numerically with scipy.optimize.brentq.
    Method searches values of lambda between 0 and max_lambda (default 1e+05)
    within the maximum number of iterations, lambda_niter (default 1000).
    """
    u_old, v_old, M = svd_cross_product(X, Y, return_matrix=True)
    ite = 1
    eps = np.finfo(X.dtype).eps
    
    u = np.zeros_like(u_old)
    v = np.zeros_like(v_old)
    k = len(x_ind) + 1
    l = len(y_ind) + 1
    x_penalty = np.zeros(k)
    y_penalty = np.zeros(l)
    x_range = [range(x_ind[i], x_ind[i+1]) for i in range(k)]
    y_range = [range(y_ind[i], y_ind[i+1]) for i in range(l)]
    # Inner loop of sgPLS
    while True:      
        # 1.1 Calculate M_v : the X projections
        M_v = np.dot(M, v_old)
        # 1.2 Calculate contribution of X groups to M_v
        for group in range(k):
            arr = np.array(M_v[x_range[group]])
            x_penalty[group] = brentq(
                    _lambda_quadratic,
                    a=0, b=max_lambda,
                    args=(arr, alpha_x),
                    xtol=lambda_tol,
                    maxiter=lambda_niter)
        # 1.3 Find lambda_x : the X penalty
        if x_group == 0:
            lambda_x = sorted(np.absolute(x_penalty))[0] - 1
        else:
            lambda_x = sorted(np.absolute(x_penalty))[x_group]        
        # Lambda must exceed a particular threshold for penalisation.
        # Therefore, it is sufficient to subtract 1 to break the condition and
        # apply penalisation.
        # See [Benoit Liquet 2015], criterion (10) and criterion (16).
        # 1.4 Update u : the X weights
        for group in range(k):
            arr = np.array(M_v[x_range[group]])
            u[x_range[group]] = _sparse_group_thresholding(
                    arr, lambda_x, x_penalty[group], alpha_x)     
        # 1.5 Normalise u
        u /= np.sqrt(np.dot(u.T, u)) + eps
        
        # 2.1 Calculate M_u : the Y projections
        M_u = np.dot(M.T, u)
        # 2.2 Calculate contribution of Y groups to M_u
        for group in range(l):
            arr = np.array(M_u[y_range[group]])
            y_penalty[group] = brentq(
                    _lambda_quadratic,
                    a=0, b=max_lambda,
                    args=(arr, alpha_y),
                    xtol=lambda_tol,
                    maxiter=lambda_niter)
        # 2.3 Find lambda_y : the Y penalty
        if y_group == 0:
            lambda_y = sorted(np.absolute(y_penalty))[0] - 1
        else:
            lambda_y = sorted(np.absolute(y_penalty))[y_group]       
        # 2.4 Update v : the Y weights
        for group in range(l):
            arr = np.array(M_u[y_range[group]])
            v[y_range[group]] = _sparse_group_thresholding(
                    arr, lambda_y, y_penalty[group], alpha_y)      
        # 2.5 Normalise u
        if norm_y_weights:
            v /= np.sqrt(np.dot(v.T, v)) + eps        
        
        u_diff = u - u_old
        v_diff = v - v_old
        if np.dot(u_diff.T, u_diff) < tol and np.dot(v_diff.T, v_diff) < tol:
            break
        if ite == max_iter:
            warnings.warn('Maximum number of iterations reached',
                          ConvergenceWarning)
            break
        u_old = u
        v_old = v
        ite += 1
    return u, v, ite


class _PLS(TransformerMixin, RegressorMixin, MultiOutputMixin, BaseEstimator,
           metaclass=ABCMeta):
    """Partial Least Squares (PLS)
    
    This class implements the generic PLS algorithm, constructors' parameters
    allow to obtain a specific implementation such as:
        
    - PLS2 regression, i.e., PLS 2 blocks, mode A, with asymmetric deflation
      and unnormalized y weights such as defined by [Tenenhaus 1998] p. 132.
      With univariate response it implements PLS1.
      
    - PLS canonical, i.e., PLS 2 blocks, mode A, with symmetric deflation and
      normalized y weights such as defined by [Tenenhaus 1998] (p. 132) and
      [Wegelin et al. 2000]. This parametrization implements the original Wold
      algorithm.
      
    We use the terminology defined by [Wegelin et al. 2000].
    This implementation uses the PLS Wold 2 blocks algorithm based on two
    nested loops:
        (i) The outer loop iterate over components.
        (ii) The inner loop estimates the weights vectors. This can be done
        with two algo. (a) the inner loop of the original NIPALS algo. or (b) a
        SVD on residuals cross-covariance matrices.
        
    n_components : int, number of components to keep. (default 2).
    
    scale : boolean, scale data? (default True)
    
    deflation_mode : str, "canonical" or "regression". See notes.
    
    mode : "A" classical PLS and "B" CCA. See notes.
    
    norm_y_weights : boolean, normalize Y weights to one? (default False)
    
    algorithm : string, "nipals" or "svd"
        The algorithm used to estimate the weights. It will be called
        n_components times, i.e. once for each iteration of the outer loop.
        
    max_iter : int (default 500)
        The maximum number of iterations
        of the NIPALS inner loop (used only if algorithm="nipals")
        
    tol : non-negative real, default 1e-06
        The tolerance used in the iterative algorithm.
        
    copy : boolean, default True
        Whether the deflation should be done on a copy. Let the default
        value to True unless you don't care about side effects.
        
    Attributes
    ----------
    x_weights_ : array, [p, n_components]
        X block weights vectors.
        
    y_weights_ : array, [q, n_components]
        Y block weights vectors.
        
    x_loadings_ : array, [p, n_components]
        X block loadings vectors.
        
    y_loadings_ : array, [q, n_components]
        Y block loadings vectors.
        
    x_scores_ : array, [n_samples, n_components]
        X scores.
        
    y_scores_ : array, [n_samples, n_components]
        Y scores.
        
    x_rotations_ : array, [p, n_components]
        X block to latents rotations.
        
    y_rotations_ : array, [q, n_components]
        Y block to latents rotations.
        
    x_mean_ : array, [p]
        X mean for each predictor.
        
    y_mean_ : array, [q]
        Y mean for each response variable.
        
    x_std_ : array, [p]
        X standard deviation for each predictor.
        
    y_std_ : array, [q]
        Y standard deviation for each response variable.
        
    coef_ : array, [p, q]
        The coefficients of the linear model: ``Y = X coef_ + Err``
        
    n_iter_ : array-like
        Number of iterations of the NIPALS inner loop for each
        component. Not useful if the algorithm given is "svd".
        
    References
    ----------
    
    Jacob A. Wegelin. A survey of Partial Least Squares (PLS) methods, with
    emphasis on the two-block case. Technical Report 371, Department of
    Statistics, University of Washington, Seattle, 2000.
    
    In French but still a reference:
    Tenenhaus, M. (1998). La regression PLS: theorie et pratique. Paris:
    Editions Technic.
    
    See also
    --------
    PLSCanonical
    PLSRegression
    """
    model = "pls"

    @abstractmethod
    def __init__(self, n_components=2, scale=True, deflation_mode="regression",
                 mode="A", algorithm="nipals", norm_y_weights=False,
                 max_iter=500, tol=1e-06, copy=True):
        self.n_components = n_components
        self.deflation_mode = deflation_mode
        self.mode = mode
        self.norm_y_weights = norm_y_weights
        self.scale = scale
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy

    def fit(self, X, Y):
        """Fit model to data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.
            
        Y : array-like of shape (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.
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
        # sPLS model: sparsity parameters
        if self.model == "spls":
            self.x_vars = pls_array(
                    array=self.x_vars, min_length=self.n_components,
                    max_length=self.n_components, min_features=0,
                    max_features=p)                
            self.y_vars = pls_array(
                    array=self.y_vars, min_length=self.n_components,
                    max_length=self.n_components, min_features=0,
                    max_features=q)
            
            x_sparsity = sparsity_conversion(self.x_vars, p)
            y_sparsity = sparsity_conversion(self.y_vars, q)
        
        # gPLS/sgPLS model: sparsity parameters and blocking inputs
        if self.model in ("gpls", "sgpls"):
            self.x_block = pls_blocks(self.x_block, 0, p)
            self.y_block = pls_blocks(self.y_block, 0, q)
            k = len(self.x_block) + 1
            l = len(self.y_block) + 1
            
            self.x_groups = pls_array(
                    array=self.x_groups, min_length=self.n_components,
                    max_length=self.n_components, min_features=0,
                    max_features=k)
            self.y_groups = pls_array(
                    array=self.y_groups, min_length=self.n_components,
                    max_length=self.n_components, min_features=0,
                    max_features=l)
            
            x_ind = np.insert(self.x_block, (0, k-1), (0, p))
            y_ind = np.insert(self.y_block, (0, l-1), (0, q))
            x_sparsity = sparsity_conversion(self.x_groups, k)
            y_sparsity = sparsity_conversion(self.y_groups, l)
        
        # sgPLS model: group sparsity mixin parameters
        if self.model == "sgpls":
            self.alpha_x = pls_array(
                    array=self.alpha_x, min_length=self.n_components,
                    max_length=self.n_components, min_features=0,
                    max_features=1)
            self.alpha_y = pls_array(
                    array=self.alpha_y, min_length=self.n_components,
                    max_length=self.n_components, min_features=0,
                    max_features=1)

        if self.n_components < 1 or self.n_components > p:
            raise ValueError('Invalid number of components: %d' %
                             self.n_components)
        if self.algorithm not in ("svd", "nipals", None):
            raise ValueError("Got algorithm %s when only 'svd', "
                             "'nipals' and None are known" % self.algorithm)
        if self.algorithm == "svd" and self.mode == "B":
            raise ValueError('Incompatible configuration: mode B is not '
                             'implemented with svd algorithm')
        if self.deflation_mode not in ["canonical", "regression"]:
            raise ValueError('The deflation mode is unknown')
        if self.algorithm == None and self.model == "pls":
            raise ValueError("Incompatible configuration: only 'svd' and "
                             "'nipals' can be implemented with PLS model")
        # Scale (in place)
        X, Y, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = (
            _center_scale_xy(X, Y, self.scale))
        # Residuals (deflated) matrices
        Xk = X
        Yk = Y
        # Results matrices
        self.x_scores_ = np.zeros((n, self.n_components))
        self.y_scores_ = np.zeros((n, self.n_components))
        self.x_weights_ = np.zeros((p, self.n_components))
        self.y_weights_ = np.zeros((q, self.n_components))
        self.x_loadings_ = np.zeros((p, self.n_components))
        self.y_loadings_ = np.zeros((q, self.n_components))
        self.n_iter_ = []

        # Outer loop, over components
        if self.model == "pls":
            Y_eps = np.finfo(Yk.dtype).eps
        for k in range(self.n_components):
            if np.all(np.dot(Yk.T, Yk) < np.finfo(np.double).eps):
                # Yk constant
                warnings.warn('Y residual constant at iteration %s' % k)
                break
            # 1) weights estimation (inner loop)
            # -----------------------------------
            if self.model == "pls":
                if self.algorithm == "nipals":
                    # Replace columns that are all close to zero with zeros
                    Yk_mask = np.all(np.abs(Yk) < 10 * Y_eps, axis=0)
                    Yk[:, Yk_mask] = 0.0
            
                    x_weights, y_weights, n_iter_ = \
                        _nipals_twoblocks_inner_loop(
                            X=Xk, Y=Yk, mode=self.mode, max_iter=self.max_iter,
                            tol=self.tol, norm_y_weights=self.norm_y_weights)
                elif self.algorithm == "svd":
                    x_weights, y_weights = svd_cross_product(
                            X=Xk, Y=Yk, return_matrix=False)
            if self.model == "spls":
                x_weights, y_weights, n_iter = \
                    _spls_inner_loop(
                            X=Xk, Y=Yk,
                            x_var=x_sparsity[k], y_var=y_sparsity[k],
                            max_iter=self.max_iter, tol=self.tol,
                            norm_y_weights=self.norm_y_weights)
            if self.model == "gpls":
                x_weights, y_weights, n_iter = \
                    _gpls_inner_loop(
                            X=Xk, Y=Yk,
                            x_group=x_sparsity[k], y_group=y_sparsity[k],
                            x_ind=x_ind, y_ind=y_ind,
                            max_iter=self.max_iter, tol=self.tol,
                            norm_y_weights=self.norm_y_weights)
            if self.model == "sgpls":
                x_weights, y_weights, n_iter = \
                    _sgpls_inner_loop(
                            X=Xk, Y=Yk,
                            x_group=x_sparsity[k], y_group=y_sparsity[k],
                            x_ind=x_ind, y_ind=y_ind,
                            alpha_x=self.alpha_x[k], alpha_y=self.alpha_y[k],
                            max_iter=self.max_iter, tol=self.tol,
                            norm_y_weights=self.norm_y_weights,
                            lambda_tol=self.lambda_tol,
                            max_lambda=self.max_lambda,
                            lambda_niter=self.lambda_niter)
            try:
                self.n_iter_.append(n_iter_)            
            except NameError:
                pass
            # Forces sign stability of x_weights and y_weights
            # Sign undeterminacy issue from svd if algorithm == "svd"
            # and from platform dependent computation if algorithm == 'nipals'
            x_weights, y_weights = svd_flip(x_weights, y_weights.T)
            y_weights = y_weights.T
            # compute scores
            x_scores = np.dot(Xk, x_weights)
            if self.norm_y_weights:
                y_ss = 1
            else:
                y_ss = np.dot(y_weights.T, y_weights)
            y_scores = np.dot(Yk, y_weights) / y_ss
            # test for null variance
            if np.dot(x_scores.T, x_scores) < np.finfo(np.double).eps:
                warnings.warn('X scores are null at iteration %s' % k)
                break
            # 2) Deflation (in place)
            # ----------------------
            # Possible memory footprint reduction may done here: in order to
            # avoid the allocation of a data chunk for the rank-one
            # approximations matrix which is then subtracted to Xk, we suggest
            # to perform a column-wise deflation.
            #
            # - regress Xk's on x_score
            x_loadings = np.dot(Xk.T, x_scores) / np.dot(x_scores.T, x_scores)
            # - subtract rank-one approximations to obtain remainder matrix
            Xk -= np.dot(x_scores, x_loadings.T)
            if self.deflation_mode == "canonical":
                # - regress Yk's on y_score, then subtract rank-one approx.
                y_loadings = (np.dot(Yk.T, y_scores)
                              / np.dot(y_scores.T, y_scores))
                Yk -= np.dot(y_scores, y_loadings.T)
            if self.deflation_mode == "regression":
                # - regress Yk's on x_score, then subtract rank-one approx.
                y_loadings = (np.dot(Yk.T, x_scores)
                              / np.dot(x_scores.T, x_scores))
                Yk -= np.dot(x_scores, y_loadings.T)
            # 3) Store weights, scores and loadings # Notation:
            self.x_scores_[:, k] = x_scores.ravel()  # T
            self.y_scores_[:, k] = y_scores.ravel()  # U
            self.x_weights_[:, k] = x_weights.ravel()  # W
            self.y_weights_[:, k] = y_weights.ravel()  # C
            self.x_loadings_[:, k] = x_loadings.ravel()  # P
            self.y_loadings_[:, k] = y_loadings.ravel()  # Q
        # Such that: X = TP' + Err and Y = UQ' + Err

        # 4) rotations from input space to transformed space (scores)
        # T = X W(P'W)^-1 = XW* (W* : p x k matrix)
        # U = Y C(Q'C)^-1 = YC* (W* : q x k matrix)
        self.x_rotations_ = np.dot(
            self.x_weights_,
            pinv2(np.dot(self.x_loadings_.T, self.x_weights_),
                  check_finite=False))
        if Y.shape[1] > 1:
            self.y_rotations_ = np.dot(
                self.y_weights_,
                pinv2(np.dot(self.y_loadings_.T, self.y_weights_),
                      check_finite=False))
        else:
            self.y_rotations_ = np.ones(1)

        if True or self.deflation_mode == "regression":
            # FIXME what's with the if?
            # Estimate regression coefficient
            # Regress Y on T
            # Y = TQ' + Err,
            # Then express in function of X
            # Y = X W(P'W)^-1Q' + Err = XB + Err
            # => B = W*Q' (p x q)
            self.coef_ = np.dot(self.x_rotations_, self.y_loadings_.T)
            self.coef_ = self.coef_ * self.y_std_
        return self

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction learned on the train data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.
            
        Y : array-like of shape (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.
            
        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.
            
        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
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
        """Transform data back to its original space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of pls components.
            
        Returns
        -------
        x_reconstructed : array-like of shape (n_samples, n_features)
        
        Notes
        -----
        This transformation will only be exact if n_components=n_features
        """
        check_is_fitted(self)
        X = check_array(X, dtype=FLOAT_DTYPES)
        # From pls space to original space
        X_reconstructed = np.matmul(X, self.x_loadings_.T)

        # Denormalize
        X_reconstructed *= self.x_std_
        X_reconstructed += self.x_mean_
        return X_reconstructed

    def predict(self, X, copy=True):
        """Apply the dimension reduction learned on the train data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.
            
        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.
            
        Notes
        -----
        This call requires the estimation of a p x q matrix, which may
        be an issue in high dimensional space.
        """
        check_is_fitted(self)
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        # Normalize
        X -= self.x_mean_
        X /= self.x_std_
        Ypred = np.dot(X, self.coef_)
        return Ypred + self.y_mean_

    def fit_transform(self, X, y=None):
        """Learn and apply the dimension reduction on the train data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.
            
        y : array-like of shape (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.
            
        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        return self.fit(X, y).transform(X, y)

    def _more_tags(self):
        return {'poor_score': True}


class PLSRegression(_PLS):
    """PLS regression
    
    PLSRegression implements the PLS 2 blocks regression known as PLS2 or PLS1
    in case of one dimensional response.
    This class inherits from _PLS with mode="A", deflation_mode="regression",
    norm_y_weights=False and algorithm="nipals".
    
    Read more in the :ref:`User Guide <cross_decomposition>`.
    
    Parameters
    ----------
    n_components : int, (default 2)
        Number of components to keep.
        
    scale : boolean, (default True)
        whether to scale the data
        
    max_iter : an integer, (default 500)
        the maximum number of iterations of the NIPALS inner loop (used
        only if algorithm="nipals")
        
    tol : non-negative real
        Tolerance used in the iterative algorithm default 1e-06.
        
    copy : boolean, default True
        Whether the deflation should be done on a copy. Let the default
        value to True unless you don't care about side effect
        
    Attributes
    ----------
    x_weights_ : array, [p, n_components]
        X block weights vectors.
        
    y_weights_ : array, [q, n_components]
        Y block weights vectors.
        
    x_loadings_ : array, [p, n_components]
        X block loadings vectors.
        
    y_loadings_ : array, [q, n_components]
        Y block loadings vectors.
        
    x_scores_ : array, [n_samples, n_components]
        X scores.
        
    y_scores_ : array, [n_samples, n_components]
        Y scores.
        
    x_rotations_ : array, [p, n_components]
        X block to latents rotations.
        
    y_rotations_ : array, [q, n_components]
        Y block to latents rotations.
        
    coef_ : array, [p, q]
        The coefficients of the linear model: ``Y = X coef_ + Err``
        
    n_iter_ : array-like
        Number of iterations of the NIPALS inner loop for each
        component.
        
    Notes
    -----
    Matrices::
        
        T: x_scores_
        U: y_scores_
        W: x_weights_
        C: y_weights_
        P: x_loadings_
        Q: y_loadings_
        
    Are computed such that::
        
        X = T P.T + Err and Y = U Q.T + Err
        T[:, k] = Xk W[:, k] for k in range(n_components)
        U[:, k] = Yk C[:, k] for k in range(n_components)
        x_rotations_ = W (P.T W)^(-1)
        y_rotations_ = C (Q.T C)^(-1)
        
    where Xk and Yk are residual matrices at iteration k.
    
    `Slides explaining
    PLS <http://www.eigenvector.com/Docs/Wise_pls_properties.pdf>`_
    
    
    For each component k, find weights u, v that optimizes:
    ``max corr(Xk u, Yk v) * std(Xk u) std(Yk u)``, such that ``|u| = 1``
    
    Note that it maximizes both the correlations between the scores and the
    intra-block variances.
    
    The residual matrix of X (Xk+1) block is obtained by the deflation on
    the current X score: x_score.
    
    The residual matrix of Y (Yk+1) block is obtained by deflation on the
    current X score. This performs the PLS regression known as PLS2. This
    mode is prediction oriented.
    
    This implementation provides the same results that 3 PLS packages
    provided in the R language (R-project):
        
        - "mixOmics" with function pls(X, Y, mode = "regression")
        - "plspm " with function plsreg2(X, Y)
        - "pls" with function oscorespls.fit(X, Y)
        
    Examples
    --------
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
    >>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> pls2 = PLSRegression(n_components=2)
    >>> pls2.fit(X, Y)
    PLSRegression()
    >>> Y_pred = pls2.predict(X)
    
    References
    ----------
    
    Jacob A. Wegelin. A survey of Partial Least Squares (PLS) methods, with
    emphasis on the two-block case. Technical Report 371, Department of
    Statistics, University of Washington, Seattle, 2000.
    
    In french but still a reference:
    Tenenhaus, M. (1998). La regression PLS: theorie et pratique. Paris:
    Editions Technic.
    """

    def __init__(self, n_components=2, scale=True,
                 max_iter=500, tol=1e-06, copy=True):
        super().__init__(
            n_components=n_components, scale=scale,
            deflation_mode="regression", mode="A",
            norm_y_weights=False, max_iter=max_iter, tol=tol,
            copy=copy)


class PLSCanonical(_PLS):
    """ PLSCanonical implements the 2 blocks canonical PLS of the original Wold
    algorithm [Tenenhaus 1998] p.204, referred as PLS-C2A in [Wegelin 2000].
    
    This class inherits from PLS with mode="A" and deflation_mode="canonical",
    norm_y_weights=True and algorithm="nipals", but svd should provide similar
    results up to numerical errors.
    
    Read more in the :ref:`User Guide <cross_decomposition>`.
    
    Parameters
    ----------
    n_components : int, (default 2).
        Number of components to keep
        
    scale : boolean, (default True)
        Option to scale data
        
    algorithm : string, "nipals" or "svd"
        The algorithm used to estimate the weights. It will be called
        n_components times, i.e. once for each iteration of the outer loop.
        
    max_iter : an integer, (default 500)
        the maximum number of iterations of the NIPALS inner loop (used
        only if algorithm="nipals")
        
    tol : non-negative real, default 1e-06
        the tolerance used in the iterative algorithm
        
    copy : boolean, default True
        Whether the deflation should be done on a copy. Let the default
        value to True unless you don't care about side effect
        
    Attributes
    ----------
    x_weights_ : array, shape = [p, n_components]
        X block weights vectors.
        
    y_weights_ : array, shape = [q, n_components]
        Y block weights vectors.
        
    x_loadings_ : array, shape = [p, n_components]
        X block loadings vectors.
        
    y_loadings_ : array, shape = [q, n_components]
        Y block loadings vectors.
        
    x_scores_ : array, shape = [n_samples, n_components]
        X scores.
        
    y_scores_ : array, shape = [n_samples, n_components]
        Y scores.
        
    x_rotations_ : array, shape = [p, n_components]
        X block to latents rotations.
        
    y_rotations_ : array, shape = [q, n_components]
        Y block to latents rotations.
        
    n_iter_ : array-like
        Number of iterations of the NIPALS inner loop for each
        component. Not useful if the algorithm provided is "svd".
        
    Notes
    -----
    Matrices::
        
        T: x_scores_
        U: y_scores_
        W: x_weights_
        C: y_weights_
        P: x_loadings_
        Q: y_loadings__
        
    Are computed such that::
        
        X = T P.T + Err and Y = U Q.T + Err
        T[:, k] = Xk W[:, k] for k in range(n_components)
        U[:, k] = Yk C[:, k] for k in range(n_components)
        x_rotations_ = W (P.T W)^(-1)
        y_rotations_ = C (Q.T C)^(-1)
        
    where Xk and Yk are residual matrices at iteration k.
    
    `Slides explaining PLS
    <http://www.eigenvector.com/Docs/Wise_pls_properties.pdf>`_
    
    For each component k, find weights u, v that optimize::
        
        max corr(Xk u, Yk v) * std(Xk u) std(Yk u), such that ``|u| = |v| = 1``
        
    Note that it maximizes both the correlations between the scores and the
    intra-block variances.
    
    The residual matrix of X (Xk+1) block is obtained by the deflation on the
    current X score: x_score.
    
    The residual matrix of Y (Yk+1) block is obtained by deflation on the
    current Y score. This performs a canonical symmetric version of the PLS
    regression. But slightly different than the CCA. This is mostly used
    for modeling.
    
    This implementation provides the same results that the "plspm" package
    provided in the R language (R-project), using the function plsca(X, Y).
    Results are equal or collinear with the function
    ``pls(..., mode = "canonical")`` of the "mixOmics" package. The difference
    relies in the fact that mixOmics implementation does not exactly implement
    the Wold algorithm since it does not normalize y_weights to one.
    
    Examples
    --------
    >>> from sklearn.cross_decomposition import PLSCanonical
    >>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
    >>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> plsca = PLSCanonical(n_components=2)
    >>> plsca.fit(X, Y)
    PLSCanonical()
    >>> X_c, Y_c = plsca.transform(X, Y)
    
    References
    ----------
    
    Jacob A. Wegelin. A survey of Partial Least Squares (PLS) methods, with
    emphasis on the two-block case. Technical Report 371, Department of
    Statistics, University of Washington, Seattle, 2000.
    
    Tenenhaus, M. (1998). La regression PLS: theorie et pratique. Paris:
    Editions Technic.
    """

    def __init__(self, n_components=2, scale=True, algorithm="nipals",
                 max_iter=500, tol=1e-06, copy=True):
        super().__init__(
            n_components=n_components, scale=scale,
            deflation_mode="canonical", mode="A",
            norm_y_weights=True, algorithm=algorithm,
            max_iter=max_iter, tol=tol, copy=copy)
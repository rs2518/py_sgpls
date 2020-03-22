import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.linalg import pinv2

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.base import MultiOutputMixin
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.extmath import svd_flip
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.cross_decomposition.pls_ import _center_scale_xy

from .utils import svd_cross_product, sparsity_conversion
from .utils import _pls_inner_loop, _spls_inner_loop
from .utils import _gpls_inner_loop, _sgpls_inner_loop
from .utils import pls_blocks, pls_array

__all__ = ['PLSCanonical', 'PLSRegression']


class _PLS(TransformerMixin, RegressorMixin, MultiOutputMixin, BaseEstimator,
           metaclass=ABCMeta):
    """Partial Least Squares (PLS)
    
    This class implements the generic mode A PLS algorithm, constructors' 
    parameters allow to obtain a specific implementation such as:
        
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
        
    norm_y_weights : boolean, normalize Y weights to one? (default False)
    
    algorithm : string, "nipals", "svd" or "NA"
        The algorithm used to estimate the weights for PLS models only
        (PLSRegression and PLSCanonical). It will be called n_components
        times, i.e. once for each iteration of the outer loop.
        Algorithm is inactive for all other pls variants and is set to "NA".
        
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
                 algorithm="nipals", norm_y_weights=False,
                 max_iter=500, tol=1e-06, copy=True):
        self.n_components = n_components
        self.deflation_mode = deflation_mode
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
        if self.n_components < 1 or self.n_components > p:
            raise ValueError('Invalid number of components: %d' %
                             self.n_components)
        if self.algorithm not in ("svd", "nipals", "NA"):
            raise ValueError("Got algorithm %s when only 'svd', "
                             "'nipals' and None are known" % self.algorithm)
        if self.deflation_mode not in ["canonical", "regression"]:
            raise ValueError('The deflation mode is unknown')
        if self.algorithm == "NA" and self.model == "pls":
            raise ValueError("Incompatible configuration: pls model must be "
                             "implemented with either 'svd' or 'nipals'")
        if self.algorithm != "NA" and self.model != "pls":
            raise ValueError("Incompatible configuration: %s model cannot "
                             "be implemented with %s. Set algorithm to 'NA'"
                             % (self.model, self.algorithm))
            
        # Validate sparsity parameters (if any)
        if self.model == "pls":
            pass
        
        elif self.model == "spls":
            self.x_vars = pls_array(
                    array=self.x_vars, min_length=self.n_components,
                    max_length=self.n_components, min_entry=0,
                    max_entry=p)                
            self.y_vars = pls_array(
                    array=self.y_vars, min_length=self.n_components,
                    max_length=self.n_components, min_entry=0,
                    max_entry=q)
            x_sparsity = sparsity_conversion(self.x_vars, p)
            y_sparsity = sparsity_conversion(self.y_vars, q)
        
        elif self.model in ("gpls", "sgpls"):
            self.x_block, x_ind = pls_blocks(self.x_block, 0, p)
            self.y_block, y_ind = pls_blocks(self.y_block, 0, q)
            self.x_groups = pls_array(
                    array=self.x_groups, min_length=self.n_components,
                    max_length=self.n_components, min_entry=0,
                    max_entry=len(x_ind)-1)
            self.y_groups = pls_array(
                    array=self.y_groups, min_length=self.n_components,
                    max_length=self.n_components, min_entry=0,
                    max_entry=len(y_ind)-1)
            x_sparsity = sparsity_conversion(self.x_groups, len(x_ind)-1)
            y_sparsity = sparsity_conversion(self.y_groups, len(y_ind)-1)
        
            if self.model == "sgpls":
                self.alpha_x = pls_array(
                        array=self.alpha_x, min_length=self.n_components,
                        max_length=self.n_components, min_entry=0,
                        max_entry=1)
                self.alpha_y = pls_array(
                        array=self.alpha_y, min_length=self.n_components,
                        max_length=self.n_components, min_entry=0,
                        max_entry=1)
                
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
        
        # Threshold for values in Y which are zeroed out in NIPALS algorithm
        Y_eps = np.finfo(Yk.dtype).eps
        
        # Outer loop, over components            
        for k in range(self.n_components):
            if np.all(np.dot(Yk.T, Yk) < np.finfo(np.double).eps):
                # Yk constant
                warnings.warn('Y residual constant at iteration %s' % k)
                break
            # 1) weights estimation (inner loop)
            # -----------------------------------
            if self.model == "pls":
                weights = \
                    _pls_inner_loop(X=Xk, Y=Yk, algorithm=self.algorithm,
                                    mode=self.mode, max_iter=self.max_iter,
                                    tol=self.tol,
                                    norm_y_weights=self.norm_y_weights,
                                    y_eps=Y_eps)
                try:
                    x_weights, y_weights, n_iter = weights
                except ValueError:
                    x_weights, y_weights = weights
                                    
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
    This class inherits from _PLS with deflation_mode="regression",
    norm_y_weights=False and algorithm="nipals".
    
    Read more in the :ref:`User Guide <cross_decomposition>`.
    
    Parameters
    ----------
    n_components : int, (default 2)
        Number of components to keep.
        
    scale : boolean, (default True)
        whether to scale the data
        
    algorithm : string, "nipals", "svd" or "NA"
        The algorithm used to estimate the weights for PLS models only
        (PLSRegression and PLSCanonical). It will be called n_components
        times, i.e. once for each iteration of the outer loop.
        Algorithm is inactive for all other pls variants and is set to "NA".
        
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

    def __init__(self, n_components=2, scale=True, algorithm="nipals",
                 max_iter=500, tol=1e-06, copy=True):
        super().__init__(
            n_components=n_components, scale=scale,
            deflation_mode="regression", algorithm=algorithm,
            norm_y_weights=False, max_iter=max_iter, tol=tol,
            copy=copy)


class PLSCanonical(_PLS):
    """ PLSCanonical implements the 2 blocks canonical PLS of the original Wold
    algorithm [Tenenhaus 1998] p.204, referred as PLS-C2A in [Wegelin 2000].
    
    This class inherits from PLS with and deflation_mode="canonical",
    norm_y_weights=True and algorithm="nipals", but svd should provide similar
    results up to numerical errors.
    
    Read more in the :ref:`User Guide <cross_decomposition>`.
    
    Parameters
    ----------
    n_components : int, (default 2).
        Number of components to keep
        
    scale : boolean, (default True)
        Option to scale data
        
    algorithm : string, "nipals", "svd" or "NA"
        The algorithm used to estimate the weights for PLS models only
        (PLSRegression and PLSCanonical). It will be called n_components
        times, i.e. once for each iteration of the outer loop.
        Algorithm is inactive for all other pls variants and is set to "NA".
        
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
            deflation_mode="canonical", algorithm=algorithm,
            norm_y_weights=True, max_iter=max_iter, tol=tol,
            copy=copy)
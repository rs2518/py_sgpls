from abc import ABCMeta, abstractmethod

from sklearn.base import RegressorMixin, MultiOutputMixin

from ._base import _PLSBase

__all__ = ['PLSCanonical', 'PLSRegression']


class _PLS(_PLSBase, RegressorMixin, MultiOutputMixin, metaclass=ABCMeta):
    """Partial Least Squares (PLS)
    
    Base PLS class for regression problems (Mode A, regression and canonical
    variants)
    """
    model = "pls"

    @abstractmethod
    def __init__(self, n_components=2, scale=True, deflation_mode="regression",
                 algorithm="nipals", norm_y_weights=False,
                 max_iter=500, tol=1e-06, copy=True):
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
            
        Y : array-like of shape (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.
        """
        super()._fit(X, Y)
        return self
    
    def predict(self, X, copy=True):
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
        return super()._decision_function(self, X)

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
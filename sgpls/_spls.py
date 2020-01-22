from ._pls import _PLS


__all__ = ['sPLSCanonical', 'sPLSRegression']


class _sPLS(_PLS):
    """Sparse Partial Least Squares (sPLS)
        
    This class implements the sPLS algorithm.
    Sparsity is enforced via the interpretable features x_vars, y_vars
    (the number of non-zero X and Y variables contributing to the loadings
    vector for each component, respectively).
    
    Specific implementations include:
        
    - sPLS regression, i.e. sPLS with asymmetric deflation and unnormalized
      y weights. The algorithm is defined in [Lê Cao 2008] and also
      mentioned in [Benoit Liquet 2015].
    
    - sPLS canonical, i.e. sPLS with symmetric deflation and 
      unnormalized y weights. The algorithm is defined in [Lê Cao 2008].
    
    For consistency, we use the terminology defined in scikit-learn's PLS
    code. However, additional functions and comments will reference
    terminology according to [Lê Cao 2008].
    The algorithm uses a two nested-loop structure similar to that of PLS:
        (i) The outer loop iterates over components
        (ii) The inner loop estimates the penalised weights vectors. Using the
        soft-thresholding iterative equation
    
    n_components : int, number of components to keep, (default 2)
    
    x_vars : array, [n_components]
        The number of non-zero X variables in the corresponding weights vector
        for each component.
    
    y_vars : array, [n_components], (default None)
        The number of non-zero Y variables in the corresponding weights vector
        for each component.
        If None, no Y variables are penalised.
    
    scale : boolean, scale data, (default True)
    
    deflation_mode : str, "canonical" or "regression". See notes.
              
    max_iter : int, (default 500)
        The maximum number of iterations of inner loop
        
    tol : non-negative real, (default 1e-06)
        The tolerance used in the iterative algorithm.
        
    copy : boolean, (default True)
        Whether the deflation should be done on a copy. Let the default
        value to True unless you don't care about side effects.
    
    Attributes
    ----------
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
        component. Not useful if the algorithm given is "svd".
    
    References
    ----------
    
    Kim-Anh Lê Cao, Debra Rossow, Christèle Robert-Granié, Philippe Besse.
    A Sparse PLS for Variable Selection when Integrating Omics data.
    Statistical Applications in Genetics and Molecular Biology,
    De Gruyter, 2008.

    Benoît Liquet, Pierre Lafaye de Micheaux, Boris P. Hejblum,
    Rodolphe Thiébaut, Group and sparse group partial least square approaches
    applied in genomics context, Bioinformatics, Volume 32, Issue 1,
    1 January 2016, Pages 35–42, https://doi.org/10.1093/bioinformatics/btv535
    
    See also
    --------
    sPLSCanonical
    sPLSRegression
    """
    model = "spls"
    
    def __init__(self, n_components=2, x_vars, y_vars=None, scale=True,
                 deflation_mode="regression", norm_y_weights=False,
                 max_iter=500, tol=1e-06, copy=True):
        super().__init__(n_components, scale, deflation_mode, mode="A",
             algorithm=None, norm_y_weights, max_iter, tol, copy)
        self.x_vars = x_vars
        self.y_vars = y_vars


class sPLSRegression(_sPLS):
    """sPLS regression
      
    sPLSRegression inherits from _sPLS with deflation_mode="regression" and
    norm_y_weights=False. Mode B is not yet supported so the sPLS class
    is configured to mode="A" and algorithm=None.
        
    Parameters
    ----------
    n_components : int, (default 2)
        Number of components to keep.
        
    x_vars : array, [n_components]
        The number of non-zero X variables in the corresponding weights vector
        for each component.
    
    y_vars : array, [n_components], (default None)
        The number of non-zero Y variables in the corresponding weights vector
        for each component.
        If None, no Y variables are penalised.
        
    scale : boolean, (default True)
        whether to scale the data
        
    max_iter : an integer, (default 500)
        the maximum number of iterations of the NIPALS inner loop (used
        only if algorithm="nipals")
        
    tol : non-negative real, (default 1e-06)
        Tolerance used in the iterative algorithm.
        
    copy : boolean, (default True)
        Whether the deflation should be done on a copy. Let the default
        value to True unless you don't care about side effect.
        
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
    The labels given by scikit-learn's PLS script are compared to the adapted
    corresponding terminology based on [Lê Cao 2008]:
    
        Matrices::     
            T: x_scores_ (or equivalently, Ξ)
            U: y_scores_ (or equivalently, Ω)
            W: x_weights_ (or equivalently, U)
            C: y_weights_ (or equivalently, V)
            P: x_loadings_ (or equivalently, C)
            Q: y_loadings_ (or equivalently, D for regression; E for canonical)
            
    References
    ----------
    
    Kim-Anh Lê Cao, Debra Rossow, Christèle Robert-Granié, Philippe Besse.
    A Sparse PLS for Variable Selection when Integrating Omics data.
    Statistical Applications in Genetics and Molecular Biology,
    De Gruyter, 2008.

    Benoît Liquet, Pierre Lafaye de Micheaux, Boris P. Hejblum,
    Rodolphe Thiébaut, Group and sparse group partial least square approaches
    applied in genomics context, Bioinformatics, Volume 32, Issue 1,
    1 January 2016, Pages 35–42, https://doi.org/10.1093/bioinformatics/btv535
    """

    def __init__(self, n_components=2, x_vars, y_vars=None, scale=True,
                 max_iter=500, tol=1e-06, copy=True):
        super().__init__(
            n_components=n_components, x_vars=x_vars, y_vars=y_vars,
            scale=scale, deflation_mode="regression", mode="A",
            algorithm=None, norm_y_weights=False, max_iter=max_iter,
            tol=tol, copy=copy)


class sPLSCanonical(_sPLS):
    """sPLS canonical

    sPLSCanonical inherits from _sPLS with deflation_mode="canonical" and
    norm_y_weights=False. Mode B is not yet supported so the sPLS class
    is configured to mode="A" and algorithm=None.
        
    Parameters
    ----------
    n_components : int, (default 2)
        Number of components to keep.
        
    x_vars : array, [n_components]
        The number of non-zero X variables in the corresponding weights vector
        for each component.
    
    y_vars : array, [n_components], (default None)
        The number of non-zero Y variables in the corresponding weights vector
        for each component.
        If None, no Y variables are penalised.
        
    scale : boolean, (default True)
        whether to scale the data
        
    max_iter : an integer, (default 500)
        the maximum number of iterations of the NIPALS inner loop (used
        only if algorithm="nipals")
        
    tol : non-negative real, (default 1e-06)
        Tolerance used in the iterative algorithm.
        
    copy : boolean, (default True)
        Whether the deflation should be done on a copy. Let the default
        value to True unless you don't care about side effect.
        
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
    The labels given by scikit-learn's PLS script are compared to the adapted
    corresponding terminology based on [Lê Cao 2008]:
    
        Matrices::     
            T: x_scores_ (or equivalently, Ξ)
            U: y_scores_ (or equivalently, Ω)
            W: x_weights_ (or equivalently, U)
            C: y_weights_ (or equivalently, V)
            P: x_loadings_ (or equivalently, C)
            Q: y_loadings_ (or equivalently, D for regression; E for canonical)   
    
    References
    ----------
    
    Kim-Anh Lê Cao, Debra Rossow, Christèle Robert-Granié, Philippe Besse.
    A Sparse PLS for Variable Selection when Integrating Omics data.
    Statistical Applications in Genetics and Molecular Biology,
    De Gruyter, 2008.

    Benoît Liquet, Pierre Lafaye de Micheaux, Boris P. Hejblum,
    Rodolphe Thiébaut, Group and sparse group partial least square approaches
    applied in genomics context, Bioinformatics, Volume 32, Issue 1,
    1 January 2016, Pages 35–42, https://doi.org/10.1093/bioinformatics/btv535
    """

    def __init__(self, n_components=2, x_vars, y_vars=None, scale=True,
                 max_iter=500, tol=1e-06, copy=True):
        super().__init__(
            n_components=n_components, x_vars=x_vars, y_vars=y_vars,
            scale=scale, deflation_mode="canonical", mode="A",
            algorithm=None, norm_y_weights=False, max_iter=max_iter,
            tol=tol, copy=copy)
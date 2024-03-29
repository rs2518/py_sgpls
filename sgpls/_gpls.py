from ._pls import _PLS
from .utils import _check_1d


__all__ = ['gPLSCanonical', 'gPLSRegression']
   

class _gPLS(_PLS):
    """Group Partial Least Squares (gPLS)
        
    This class implements the gPLS algorithm.
    Sparsity is enforced on groups of variables defined by the user a priori,
    using the attributes x_groups and y_groups (the number of non-zero groups
    of X and Y variables contributing to the loadings vectors for each
    component, respectively).
    
    Specific implementations include:
        
    - gPLS regression, i.e. gPLS with asymmetric deflation and unnormalized
      y weights. The algorithm is defined in [Benoit Liquet 2015].
    
    - gPLS canonical, i.e. gPLS with symmetric deflation and unnormalized 
      y weights. The algorithm is defined in [Benoit Liquet 2015].
    
    For consistency, we use the terminology defined in scikit-learn's PLS
    code. However, additional functions and comments will reference
    terminology according to [Benoit Liquet 2015] (and [Lê Cao 2008] where
    appropriate).
    The algorithm uses a two nested-loop structure similar to that of PLS:
        (i) The outer loop iterates over components
        (ii) The inner loop estimates the penalised weights vectors according
        to their respective groups.
    
    n_components : int, number of components to keep, (default 2)
    
    x_block : array, [k_groups]
        An array that describes the groupings of X variables. The entries 
        correspond to the starting index of each X group.
        
    x_groups : array, [n_components]
        The number of non-zero groups of X variables (as defined by x_block)
        in the corresponding weights vector for each component.

    y_block : array, [l_groups]
        An array that describes the groupings of Y variables. The entries 
        correspond to the starting index of each Y group.
        If None, all Y variables form a single group.

    y_groups : array, [n_components], (default None)
        The number of non-zero groups of Y variables (as defined by y_block)
        in the corresponding weights vector for each component.
        If None, no Y groups are penalised.
        
    scale : boolean, scale data, (default True)
    
    deflation_mode : str, "canonical" or "regression". See notes.
              
    max_iter : int, (default 500)
        The maximum number of iterations of inner loop
        
    tol : non-negative real, (default 1e-06)
        Tolerance used in the iterative algorithm.
        
    copy : boolean, (default True)
        Whether the deflation should be done on a copy. Let the default
        value to True unless you don't care about side effect.
    
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
        Number of iterations of the inner loop for each component.
    
    Notes
    -----
    The x_block and y_block attributes describe the grouping structures of
    X and Y variables.
    Thus, the ith entry gives the starting index for group (i + 1).
    The total number of groups is equal to the array length + 1.
    E.g.
        Let X_1, ..., X_p denote the variables in the X data matrix.
        Suppose that x_block is
            array([3, 7, 10])
        
        The X variables are sectioned into 4 groups as follows:
            1: X_1 to X_2
            2: X_3 to X_6
            3: X_7 to X_9
            4: X_10 to X_p
    
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
    gPLSCanonical
    gPLSRegression
    """
    model = "gpls"

    def __init__(self, x_block, x_groups, y_block=None, y_groups=None,
                 n_components=2, scale=True, deflation_mode="regression",
                 norm_y_weights=False, max_iter=500, tol=1e-06, copy=True):
        super().__init__(n_components=n_components, scale=scale,
                         deflation_mode=deflation_mode, algorithm="NA",
                         norm_y_weights=norm_y_weights,
                         max_iter=max_iter, tol=tol, copy=copy)        
        self.x_block = _check_1d(x_block)
        self.x_groups = _check_1d(x_groups)
        if y_block is None:
            self.y_block = y_block
        else:
            self.y_block = _check_1d(y_block)
        if y_groups is None:
            self.y_groups = y_groups
        else:
            self.y_groups = _check_1d(y_groups)
        
        
class gPLSRegression(_gPLS):
    """gPLS regression
    
    gPLSRegression inherits from _gPLS with deflation_mode="regression",
    norm_y_weights=False and with algorithm="NA".
        
    Parameters
    ----------
    n_components : int, (default 2)
        Number of components to keep.
        
    x_block : array, [k_groups]
        An array that describes the groupings of X variables. The entries 
        correspond to the starting index of each X group.
        
    x_groups : array, [n_components]
        The number of non-zero groups of X variables (as defined by x_block)
        in the corresponding weights vector for each component.

    y_block : array, [l_groups]
        An array that describes the groupings of Y variables. The entries 
        correspond to the starting index of each Y group.
        If None, all Y variables form a single group.

    y_groups : array, [n_components], (default None)
        The number of non-zero groups of Y variables (as defined by y_block)
        in the corresponding weights vector for each component.
        If None, no Y groups are penalised.
        
    scale : boolean, scale data, (default True)
    
    deflation_mode : str, "canonical" or "regression". See notes.
              
    max_iter : int, (default 500)
        The maximum number of iterations of inner loop
        
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
        Number of iterations of the inner loop for each component.
        
    Notes
    -----
    The x_block and y_block attributes describe the grouping structures of
    X and Y variables.
    Thus, the ith entry gives the starting index for group (i + 1).
    The total number of groups is equal to the array length + 1.
    E.g.
        Let X_1, ..., X_p denote the variables in the X data matrix.
        Suppose that x_block is
            array([3, 7, 10])
        
        The X variables are sectioned into 4 groups as follows:
            1: X_1 to X_2
            2: X_3 to X_6
            3: X_7 to X_9
            4: X_10 to X_p

    
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

    def __init__(self, x_block, x_groups, y_block=None, y_groups=None,
                 n_components=2, scale=True, max_iter=500, tol=1e-06,
                 copy=True):
        super().__init__(
            x_block, x_groups, y_block=y_block, y_groups=y_groups,
            n_components=n_components, scale=scale,
            deflation_mode="regression", norm_y_weights=True,
            max_iter=max_iter, tol=tol, copy=copy)


class gPLSCanonical(_gPLS):
    """gPLS canonical

    gPLSCanonical inherits from _gPLS with deflation_mode="canonical",
    norm_y_weights=True and with algorithm="NA".
        
    Parameters
    ----------
    n_components : int, (default 2)
        Number of components to keep.
        
    x_block : array, [k_groups]
        An array that describes the groupings of X variables. The entries 
        correspond to the starting index of each X group.
        
    x_groups : array, [n_components]
        The number of non-zero groups of X variables (as defined by x_block)
        in the corresponding weights vector for each component.

    y_block : array, [l_groups]
        An array that describes the groupings of Y variables. The entries 
        correspond to the starting index of each Y group.
        If None, all Y variables form a single group.

    y_groups : array, [n_components], (default None)
        The number of non-zero groups of Y variables (as defined by y_block)
        in the corresponding weights vector for each component.
        If None, no Y groups are penalised.
        
    scale : boolean, scale data, (default True)
    
    deflation_mode : str, "canonical" or "regression". See notes.
              
    max_iter : int, (default 500)
        The maximum number of iterations of inner loop
        
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
        Number of iterations of the inner loop for each component.
        
    Notes
    -----
    The x_block and y_block attributes describe the grouping structures of
    X and Y variables.
    Thus, the ith entry gives the starting index for group (i + 1).
    The total number of groups is equal to the array length + 1.
    E.g.
        Let X_1, ..., X_p denote the variables in the X data matrix.
        Suppose that x_block is
            array([3, 7, 10])
        
        The X variables are sectioned into 4 groups as follows:
            1: X_1 to X_2
            2: X_3 to X_6
            3: X_7 to X_9
            4: X_10 to X_p
    
    
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

    def __init__(self, x_block, x_groups, y_block=None, y_groups=None,
                 n_components=2, scale=True, max_iter=500, tol=1e-06,
                 copy=True):
        super().__init__(
            x_block, x_groups, y_block=y_block, y_groups=y_groups,
            n_components=n_components, scale=scale,
            deflation_mode="canonical", norm_y_weights=True,
            max_iter=max_iter, tol=tol, copy=copy)
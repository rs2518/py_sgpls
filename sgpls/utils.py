import warnings

import numpy as np
from scipy.linalg import svd, pinv2
from scipy.optimize import brentq

from sklearn.utils import check_array
from sklearn.exceptions import ConvergenceWarning


def svd_cross_product(X, Y, return_matrix=False):
    """Returns singular vectors from matrix product of X.T and Y.
    
    NOTE: 'return_matrix' parameter added to return matrix product of X.T 
    and Y if True.
    """
    C = np.dot(X.T, Y)
    U, s, Vh = svd(C, full_matrices=False)
    u = U[:, [0]]
    v = Vh.T[:, [0]]
    if return_matrix:
        return u, v, C
    else:
        return u, v


def sparsity_conversion(array, n):
    """Sparsity conversion function.
    
    Converts number of non-zero variables/groups of variables to number of
    variables/groups of variables to be penalised.
    None is converted into vector of zeros for no penalisation.
    """
    if array is None:
        return np.zeros(n)
    else:
        return n - array


def _soft_thresholding(y, lambda_k):
    """Soft-thresholding function.
    
    Recursive formula which penalises the PLS weights. Derived from the
    LASSO penalisation of the objective function.
    """
    return np.sign(y) * np.clip(np.abs(y) - lambda_k, a_min=0, a_max=None)


def _group_thresholding(y, lambda_k, penalty):
    """Group thresholding function for gPLS.
    
    Recursive formula which penalises groups of PLS weights (defined by the
    user). Derived from the group LASSO penalisation of the objective function.    
    """
    return np.array(y) * np.clip(1 - (lambda_k / penalty),
                    a_min = 0, a_max = None)


def _lambda_quadratic(y, lambda_k, alpha):
    """Lambda quadratic for sgPLS.
    
    Quadratic equation with respect to lambda for given mixing parameter
    alpha. The roots give the threshold for lambda such that values
    of lambda exceeding this threshold will produce non-zero PLS weights
    """
    g = _soft_thresholding(y, lambda_k * alpha / 2)
    return np.dot(g.T, g) - len(g) * (lambda_k * (1 - alpha)) ** 2


def _sparse_group_thresholding(y, lambda_k, penalty, alpha):
    """Sparse group thresholding function for sgPLS.
    
    Recursive formula which penalises groups of PLS weights defined by the
    user (defined by the user) whilst penalising individual weights.
    Derived from a combination of LASSO and group LASSO penalisation of the
    objective function.
    """
    if penalty < lambda_k:
        pass
    else:
        g = _soft_thresholding(y, lambda_k * alpha / 2)
        c = 1 - lambda_k * (1 - alpha) * np.sqrt(len(g))/np.dot(g.T, g)
        return c/2 * g


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



def _check_1d(array):
    """Check if input is a non-empty 1D array
    
    Uses scikit-learn check_array for input validation. (See scikit_learn
    API reference for more information)
    
    NOTE: Empty numpy arrays are still 1D arrays but have a length of 0
    """
    # Input validation
    array_converted = check_array(array, ensure_2d=False)
    
    # Check if 1D
    if array_converted.ndim not in (1, 2) :
        raise ValueError("Shape of array is invalid: \n %s.\n"
                         "Array must be 1-dimensional"
                         % array_converted)
    elif array_converted.ndim == 2 and 1 not in array_converted.shape:
        raise ValueError("Shape of array is invalid: \n %s.\n"
                         "Array must be 1-dimensional"
                         % array_converted)
    else:
        array_converted = array_converted.flatten()
        
    return array_converted


def _validate_block(array):
    """Additinal checks for blocking array (used in gPLS and sgPLS).
    
    Ensures that array entries are in ascending order and checks that there 
    are no repeated entries.
    """
    # Check if in ascending order
    if any(array != sorted(array)):
        raise ValueError("Invalid blocking structure: \n %s.\n"
                         "Entries must be in ascending order"
                         % array)
    
    # Check for repeated entries
    if len(array) != len(np.unique(array)):
        raise ValueError("Invalid blocking structure: \n %s.\n"
                         "Cannot have repeated entries"
                         % array)
    
    return array


def pls_array(array, max_length, min_length, max_entry, min_entry=0):
    """Validates input arguments for sparse extensions of PLS.
    
    Combination of _check_1d plus additional checks.
    Ensures that the length of array is between min_length and max_length
    (inclusive) and that all entries are between min_entry and max_entry
    (See parameters for details).
    
    Parameters
    ----------
    array : array
        Input to be validated.
        
    max_length : int
        Maximum length that the input array can be.
        
    min_length : int
        Minimum length that the input array can be (must be greater than 0).
        
    max_entry : int
        Maximum value of the entries allowed in the input array.
        This corresponds with the maximum number of features that can be 
        possibly selected from the data matrices.
        
    min_entry : int (default = 0)
        Minimum value of the entries allowed in the input array.
        This corresponds with the minimum number of features that can be 
        possibly selected from the data matrices.
        
    Returns
    -------
    array_converted : array
        Validated array. ValueError raised if array is invalid.
    """
    if array is None:
        return None
    
    else:
        # Validate array
        array_converted = _check_1d(array)
        
        # Check that length of array is between minimum and maximum length
        if len(array_converted) < min_length:
            raise ValueError("Length of array is invalid: \n %s.\n" 
                             "Length is %d but minimum length is %d"
                             % (array_converted, len(array_converted),
                                min_length))
        if len(array_converted) > max_length:
            raise ValueError("Length of array is invalid: \n %s.\n" 
                             "Length is %d but maximum length is %d"
                             % (array_converted, len(array_converted),
                                max_length))
        
        # Check that all entries are between minimum and maximum value
        if any((array_converted < min_entry) |
               (array_converted > max_entry)):
            raise ValueError("Invalid entry in array: \n %s.\n" 
                             "All entries must be between %d and %d"
                             % (array_converted, min_entry,
                                max_entry))
            
        return array_converted


def pls_blocks(array, max_entry, min_entry=0):
    """Validate blocking inputs for gPLS and sgPLS
    
    Combination of _pls_array, _validate_block plus additional checks.
    If 0 or max_entry appears at the end points of the array,
    a warning is raised and the indices are removed from the array.
    
    Parameters
    ----------
    array : array
        Input to be validated.
        
    max_entry : int
        Maximum value of the entries allowed in the input array.
        This corresponds with the maximum number of features that can be 
        possibly selected from the data matrices.
        
    min_entry : int (default = 0)
        Minimum value of the entries allowed in the input array (default = 0).
        This corresponds with the minimum number of features that can be 
        possibly selected from the data matrices.
        
    Returns
    -------
    array_converted : array
        Validated array. ValueError raised if array is invalid.
        Length of array must be between 1 and max_entry.
    """
    if array is None:
        return None
    
    else:
        # Input validation
        array_converted = pls_array(array, min_length=1,
                                     max_length=max_entry,
                                     min_entry=0,
                                     max_entry=max_entry)
        array_converted = _validate_block(array_converted)
        
        # Remove 0 and/or max_entry
        if array_converted[0] == 0:
            array_converted = array_converted[1:]
            message = "'%d' index removed from blocking array" % 0
            warnings.warn(message)
        
        if array_converted[-1] == max_entry:
            array_converted = array_converted[:-1]
            message = "'%d' index removed from blocking array" % max_entry
            warnings.warn(message)
            
        return array_converted
import warnings

import numpy as np
from scipy.linalg import svd
from scipy.optimize import brentq

from sklearn.cross_decomposition._pls import _nipals_twoblocks_inner_loop
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
    eps = np.finfo(float).eps ** 0.5
    a = np.abs(y) - lambda_k
    if a < eps:
        a = 0
    return np.sign(y) * np.clip(a, a_min=0, a_max=None)


def _group_thresholding(y, lambda_k, penalty):
    """Group thresholding function for gPLS.
    
    Recursive formula which penalises groups of PLS weights (defined by the
    user). Derived from the group LASSO penalisation of the objective function.    
    """
    eps = np.finfo(float).eps ** 0.5
    a = 1 - (lambda_k / penalty)
    if a < eps:
        a = 0
    return np.array(y) * np.clip(a, a_min = 0, a_max = None)


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


def _pls_inner_loop(X, Y, y_eps, algorithm="nipals", max_iter=500,
                    tol=1e-06, norm_y_weights=False):
    """Inner loop for the tuning of the PLS algorithm.
    
    If algorithm = "nipals", the NIPALS implementation is used to estimate the
    weights (See sklearn.cross_decomposition._pls for details).
    If algorithm = "svd", the SVD implementation is used to estimate the
    weights. SVD is not an iterative method and therefore does not return the
    number of iterations.
    """
    if algorithm == "nipals":
        # Replace columns that are all close to zero with zeros
        Y_mask = np.all(np.abs(Y) < 10 * y_eps, axis=0)
        Y[:, Y_mask] = 0.0
        # NIPALS algorithm
        u, v, ite = \
            _nipals_twoblocks_inner_loop(X=X, Y=Y, mode="A",
                                         max_iter=max_iter, tol=tol,
                                         norm_y_weights=norm_y_weights)
    
    elif algorithm == "svd":
        # SVD algorithm (non-iterative method)
        u, v = svd_cross_product(X=X, Y=Y, return_matrix=False)
        ite = None
    
    return u, v, ite


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
            lambda_x = sorted(np.absolute(M_v))[x_var - 1]
        # The number of non-zero X loadings gives the appropriate value
        # for penalisation of X variables.
        # 1.3 Update u : the X weights
        u = _soft_thresholding(M_v, lambda_x)
        if np.dot(u.T, u) < eps:
            u += eps
        # 1.4 Normalise u
        u /= np.sqrt(np.dot(u.T, u)) + eps
        
        # 2.1 Calculate M_u : the Y projections
        M_u = np.dot(M.T, u)
        # 2.2 Find lambda_y : the Y penalty
        if y_var == 0:
            lambda_y = 0
        else:
            lambda_y = sorted(np.absolute(M_u))[y_var - 1]
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
    k = len(x_ind) - 1
    l = len(y_ind) - 1
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
            lambda_x = sorted(np.absolute(x_penalty))[x_group - 1]        
        # Groups of X variables are penalised using the group thresholding
        # function and the appropriate penalty calculated from the magnitude
        # of the projections for each group.
        # 1.4 Update u : the X weights        
        for group in range(k):
            arr = np.array(M_v[x_range[group]])
            u[x_range[group]] = _group_thresholding(
                    arr, lambda_x, x_penalty[group])
        if np.dot(u.T, u) < eps:
            u += eps
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
            lambda_y = sorted(np.absolute(y_penalty))[y_group - 1]        
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
                      max_lambda=1e+05, lambda_max_iter=1000):
    """Inner loop for iterative tuning of sgPLS algorithm.
    
    Estimates PLS weights which solve the sparse group PLS objective function.
    See Benoît Liquet et al (2015) for details.  
    Lambda thresholds are solved numerically with scipy.optimize.brentq.
    Method searches values of lambda between 0 and max_lambda (default 1e+05)
    within the maximum number of iterations, lambda_max_iter (default 1000).
    """
    u_old, v_old, M = svd_cross_product(X, Y, return_matrix=True)
    ite = 1
    eps = np.finfo(X.dtype).eps
    
    u = np.zeros_like(u_old)
    v = np.zeros_like(v_old)
    k = len(x_ind) - 1
    l = len(y_ind) - 1
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
                    maxiter=lambda_max_iter)
        # 1.3 Find lambda_x : the X penalty
        if x_group == 0:
            lambda_x = sorted(np.absolute(x_penalty))[0] - 1
        else:
            lambda_x = sorted(np.absolute(x_penalty))[x_group - 1]        
        # Lambda must exceed a particular threshold for penalisation.
        # Therefore, it is sufficient to subtract 1 to break the condition for
        # applyling penalisation.
        # See [Benoit Liquet 2015], criterion (10) and criterion (16).
        # 1.4 Update u : the X weights
        for group in range(k):
            arr = np.array(M_v[x_range[group]])
            u[x_range[group]] = _sparse_group_thresholding(
                    arr, lambda_x, x_penalty[group], alpha_x)
        if np.dot(u.T, u) < eps:
            u += eps
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
                    maxiter=lambda_max_iter)
        # 2.3 Find lambda_y : the Y penalty
        if y_group == 0:
            lambda_y = sorted(np.absolute(y_penalty))[0] - 1
        else:
            lambda_y = sorted(np.absolute(y_penalty))[y_group - 1]       
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
    """Check if input is a non-empty 1D array. If array is None,
    None is returned.
    
    Uses scikit-learn check_array for input validation. (See scikit_learn
    API reference for more information)
    
    NOTE: Empty numpy arrays are still 1D arrays but have a length of 0
    """
    if array is None:
        pass
    
    # Input validation
    else:
        array_converted = check_array(array, ensure_2d=False)
        
        # Check if 1D
        if array_converted.ndim not in (1, 2):
            raise ValueError("Shape of array is invalid: \n%s\n"
                             "Array must be 1-dimensional"
                             % array_converted)
        elif array_converted.ndim == 2 and 1 not in array_converted.shape:
            raise ValueError("Shape of array is invalid: \n%s\n"
                             "Array must be 1-dimensional"
                             % array_converted)
        else:
            array_converted = array_converted.flatten() 
        
        return array_converted


def _validate_block(array):
    """Additional checks for blocking array (used in gPLS and sgPLS).
    
    Ensures that array entries are in ascending order and checks that there 
    are no repeated entries.
    """
    # Check if in ascending order
    if any(array != sorted(array)):
        raise ValueError("Invalid blocking structure: \n%s\n"
                         "Entries must be in ascending order"
                         % array)
    
    # Check for repeated entries
    if len(array) != len(np.unique(array)):
        raise ValueError("Invalid blocking structure: \n%s\n"
                         "Cannot have repeated entries"
                         % array)
    
    return array


def pls_array(array, max_length=None, min_length=1, max_entry=None,
              min_entry=0):
    """Validates input arguments for sparse extensions of PLS.
    
    Ensures that the length of array is between min_length and max_length
    (inclusive) and that all entries are between min_entry and max_entry
    (See parameters for details).
    
    Parameters
    ----------
    array : array
        Input to be validated. If array is None, other inputs are inactive.
        
    max_length : int (default = None)
        Maximum length that the input array can be. Setting to None disables
        check.
        
    min_length : int (default = 1)
        Minimum length that the input array can be. Setting to None disables
        check.
        
    max_entry : int (default = None)
        Maximum value of the entries allowed in the input array.
        This corresponds with the maximum number of features that can be 
        possibly selected from the data matrices. Setting to None disables
        check.
        
    min_entry : int (default = 0)
        Minimum value of the entries allowed in the input array.
        This corresponds with the minimum number of features that can be 
        possibly selected from the data matrices. Setting to None disables
        check.

    Returns
    -------
    array_converted : array
        Validated array. ValueError raised if array is invalid. If array is
        None, None is returned.
    """
    if array is None:
        pass
    
    else:
        # Check that length of array is between minimum and maximum length
        if min_length is not None and len(array) < min_length:
            raise ValueError("Length of array is invalid: \n%s\n" 
                             "Length is %d but minimum length is %d"
                             % (array, len(array),
                                min_length))
        if max_length is not None and len(array) > max_length:
            raise ValueError("Length of array is invalid: \n%s\n" 
                             "Length is %d but maximum length is %d"
                             % (array, len(array),
                                max_length))
        
        # Check that all entries are between minimum and maximum value
        if min_entry is not None and any(array < min_entry):
            raise ValueError("Invalid entry in array: \n%s\n"
                             "Entries cannot be < %d"
                             % (array, min_entry))
        
        if max_entry is not None and any(array > max_entry):
            raise ValueError("Invalid entry in array: \n%s\n"
                             "Entries cannot be > %d"
                             % (array, max_entry))    
            
        return array


def pls_blocks(array, max_entry, min_entry=0, warn=False):
    """Validate blocking inputs for gPLS and sgPLS
    
    Combination of _validate_block, _pls_array plus additional checks.
    If min entry or max_entry appears at the end points of the array, an
    optional warning is raised and the indices are removed from the array.
    Another array with both endpoints added is also returned.
    
    Parameters
    ----------
    array : array
        Input to be validated. If array is None, None and
        np.array([min_entry, max_entry]) are returned.
        
    max_entry : int
        Maximum value of the entries allowed in the input array.
        This corresponds with the maximum number of features that can be 
        possibly selected from the data matrices. Setting to None disables
        check.
        
    min_entry : int (default = 0)
        Minimum value of the entries allowed in the input array.
        This corresponds with the minimum number of features that can be 
        possibly selected from the data matrices. Setting to None disables
        check.
        
    warn : bool (default = False)
        If True, returns warning messages to indicate whether indices have
        been removed from array.
        
    Returns
    -------
    array_converted : array
        Validated array. ValueError raised if array is invalid. If array is
        None, None is returned.
        
    ind : array
        Returns array with endpoints inserted onto both sides of
        array_converted. If array is None, np.array([min_entry, max_entry])
        is returned.
    """ 
    if array is None:
        array_converted = array
        ind = np.zeros(0)
    
    else:
        # Input validation
        array_converted = _validate_block(array)
        
        array_converted = pls_array(array_converted,
                                    max_length=max_entry,
                                    min_length=1,
                                    max_entry=max_entry,
                                    min_entry=min_entry)
        
        # Remove 0 and/or max_entry. Create warning message
        msg = 0
        if array_converted[0] == min_entry:
            array_converted = array_converted[1:]
            msg += 1
        
        if array_converted[-1] == max_entry:
            array_converted = array_converted[:-1]
            msg += 2
        ind = np.copy(array_converted)
        
        if warn and msg != 0:
            warn_messages = {1: "Removed '%d' from blocking array"
                             % min_entry,
                             2: "Removed '%d' from blocking array"
                             % max_entry,
                             3: "Removed '%d' and '%d' from blocking array"
                             % (min_entry, max_entry)}
            warnings.warn(warn_messages[msg])
    
    # Insert endpoints
    ind = np.insert(ind, (0, len(ind)), (min_entry, max_entry))
            
    return array_converted, ind
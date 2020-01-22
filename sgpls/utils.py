import warnings

import numpy as np
from scipy.linalg import svd

from sklearn.utils import check_array


def _svd_cross_product(X, Y, return_matrix=False):
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


def _center_scale_xy(X, Y, scale=True):
    """Center X, Y and scale if the scale parameter==True
        
    Returns
    -------
        X, Y, x_mean, y_mean, x_std, y_std
    """
    # center
    x_mean = X.mean(axis=0)
    X -= x_mean
    y_mean = Y.mean(axis=0)
    Y -= y_mean
    # scale
    if scale:
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = Y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        Y /= y_std
    else:
        x_std = np.ones(X.shape[1])
        y_std = np.ones(Y.shape[1])
    return X, Y, x_mean, y_mean, x_std, y_std


def _sparsity_conversion(array, n):
    """Sparsity conversion function.
    
    Converts number of non-zero variables/groups of variables to number of
    variables/groups of variables to be penalised.
    None is converted into vector of zeros for no penalisation.
    """
    if array == None:
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



def _check_1d(array):
    """Check if input is a non-empty 1D array
    
    Uses scikit-learn check_array for input validation. (See scikit_learn
    API reference for more information)
    """
    # Input validation
    array_converted = check_array(array, ensure_2d=False)
    
    # Check if 1D
    if array_converted.ndim != 1:
        raise ValueError("Shape of array is invalid: \n %s.\n" 
                         "Array must be 1-dimensional"
                         % array_converted)
    
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


def _pls_array(array, min_length, max_length, min_entry=0, max_entry):
    """Validates input arguments for sparse extensions of PLS.
    
    Combination of _check_1d plus additional checks.
    Ensures that the length of array is between min_length and max_length
    (inclusive) and that all entries are between min_entry and max_entry
    (See parameters for details).
    
    Parameters
    ----------
    array : array
        Input to be validated.
    
    min_length : int
        Minimum length that the input array can be (must be greater than 0).
    
    max_length : int
        Maximum length that the input array can be.
    
    min_entry : int
        Minimum value of the entries allowed in the input array (default = 0).
        This corresponds with the minimum number of features that can be 
        possibly selected from the data matrices.
        
    max_entry : int
        Maximum value of the entries allowed in the input array.
        This corresponds with the maximum number of features that can be 
        possibly selected from the data matrices.
        
    Returns
    -------
    array_converted : array
        Validated array. ValueError returned if array is invalid.
    """
    if array == None:
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


def _pls_blocks(array, min_entry=0, max_entry):
    """Validate blocking inputs for gPLS and sgPLS
    
    Combination of _pls_array, _validate_block plus additional checks.
    If 0 or max_entry appears at the end points of the array,
    a warning is raised and the indices are removed from the array.
    
    Parameters
    ----------
    array : array
        Input to be validated.
    
    min_entry : int
        Minimum value of the entries allowed in the input array (default = 0).
        This corresponds with the minimum number of features that can be 
        possibly selected from the data matrices.
        
    max_entry : int
        Maximum value of the entries allowed in the input array.
        This corresponds with the maximum number of features that can be 
        possibly selected from the data matrices.
        
    Returns
    -------
    array_converted : array
        Validated array. ValueError returned if array is invalid.
        Length of array must be between 1 and max_entry.
    """
    if array == None:
        return None
    
    else:
        # Input validation
        array_converted = _pls_array(array, min_length=1,
                                     max_length=max_entry,
                                     min_entry=0,
                                     max_entry=max_entry)
        array_converted = _validate_block(array_converted)
        
        # Remove 0 and/or max_entry
        if array_converted[0] == 0:
            array_converted = array_converted[1:]
            warnings.warn("'%d' index removed from blocking array" %
                          0)
        
        if array_converted[-1] == max_entry:
            array_converted = array_converted[:-1]
            warnings.warn("'%d' index removed from blocking array" %
                          max_entry)
            
        return array_converted
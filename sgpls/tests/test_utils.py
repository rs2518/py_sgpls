# =============================================================================
# TEMPORARY IMPORT. REMOVE WHEN FINALISED
# =============================================================================
# Import locally
import os, sys
directory = 'Desktop/py_sgpls'
path = os.path.join(os.path.abspath('.'), directory)
sys.path.append(path)
# =============================================================================
# =============================================================================
import numpy as np

from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_raises, assert_warns)
from sgpls.utils import sparsity_conversion
from sgpls.utils import _soft_thresholding
from sgpls.utils import _group_thresholding
from sgpls.utils import _lambda_quadratic
from sgpls.utils import _sparse_group_thresholding
from sgpls.utils import _check_1d
from sgpls.utils import _validate_block
from sgpls.utils import pls_array
from sgpls.utils import pls_blocks



def test_sparsity_conversion():
    # Test simple array
    a = np.array([0, 1, 2, 3, 4, 5, 6])
    sc = sparsity_conversion(a, 3)
    
    assert_array_equal(sc, np.array([3, 2, 1, 0, -1, -2, -3]))
    
    # Test None case
    b = None
    sc2 = sparsity_conversion(b, 5)
    
    assert_array_equal(sc2, np.array([0 ,0 ,0, 0, 0]))
    
    
def test_soft_thresholding():
    array = np.array([7, 0, -5, -2, 6, -4, -10])
    st = _soft_thresholding(array, 4)
    
    true_st = np.array([3, 0, -1, 0, 2, 0, -6])
    
    assert_array_equal(st, true_st)
    
    
def test_group_thresholding():
    array = np.array([0, -5, 6, -9, 4, 7, 2])
    
    # Case 1: lambda_k/penalty <= 1. Array is not penalised
    gt = _group_thresholding(array, 1, 2)    # lambda_k/penalty = 0.5 <= 1
    
    true_gt = np.array([0, -2.5, 3, -4.5, 2, 3.5, 1])
    
    assert_array_equal(gt, true_gt)
    
    # Case 2: lambda_k/penalty > 1. Array is penalised (zeroed out)
    gt2 = _group_thresholding(array, 2, 1)    # lambda_k/penalty = 2 > 1
    
    assert_array_equal(gt2, np.zeros(len(array)))
    
    
def test_lambda_quadratic():
    array = np.array([-0.3, 0.8, 0.9, 1.0, 0.4])
    lambdas = np.linspace(0,5,11)
    lq = np.array([_lambda_quadratic(lam, array, 1/2)
                   for lam in lambdas]).ravel()
    
    # Compare to lambda.quadra() from sgPLS package internal functions
    true_lq = np.array([2.7, 1.615625, 0.0625, -1.965, -4.5, -7.565625,
                        -11.1625, -15.29625, -20, -25.3125, -31.25])
    
    assert_array_almost_equal(lq, true_lq)
    
    
def test_sparse_group_thresholding():
    array = np.array([7, 0, -5, -2])
    sgt = _sparse_group_thresholding(array, 16, 1e+06, 1/2)
    
    true_sgt = np.array([-4.589466384, 0, 1.529822128, 0]) # R version
    # true_sgt = np.array([-0.9, 0, 0.3, 0]) # Literature version
    
    assert_array_almost_equal(sgt, true_sgt)
    
    
def test_check_1d():    
    # Check that multi-dimensional arrays with n.dim > 2 raise ValueErrors
    multi_d = np.zeros((5, 4, 3))
    assert_raises(ValueError, _check_1d, multi_d)
    
    # Check that 2D arrays raise ValueErrors
    two_d = np.zeros((5, 4))
    assert_raises(ValueError, _check_1d, two_d)
    
    # Test 1D arrays
    one_d_a = np.zeros(5)
    assert_array_equal(one_d_a, _check_1d(one_d_a))
    
    # Check that 2D arrays with shapes (n,1) or (1,n) are converted to 1D
    one_d_b = np.zeros((5, 1))
    one_d_c = np.zeros((1, 5))
    assert_array_equal(one_d_a, _check_1d(one_d_b))
    assert_array_equal(one_d_a, _check_1d(one_d_c))
    
    
def test_validate_block():
    a = np.array([0, 3, 5, 8, 9])
    assert_array_equal(a, _validate_block(a))
    
    # Check that unordered arrays and arrays with repeated entries raise
    # ValueError
    b = np.array([9, 3, 5, 8, 0])   # Swap first and last entry
    c = np.array([0, 3, 3, 5, 8, 9])    # Repeat second entry
    assert_raises(ValueError, _validate_block, b)
    assert_raises(ValueError, _validate_block, c)
    
    
def test_pls_array():
    # Case 1: valid array, min_length < max_length
    a = np.array([2, 4, 0, 3, 1, 5])
    assert_array_equal(a,
                                  pls_array(a, max_length=6, min_length=0,
                                            max_entry=5, min_entry=0))
    
    # Case 2: valid array, min_length = max_length
    assert_array_equal(a,
                                  pls_array(a, max_length=6, min_length=6,
                                            max_entry=5, min_entry=0))
    
    # Case 3: entries < min_entry. Raises ValueError
    b = np.array([-2, 4, 0, 3, 1, 5])
    assert_raises(ValueError, pls_array, b, 6, 0, 5, 0)
    
    # Case 4: entries > max_entry. Raises ValueError
    c = np.array([100, 4, 0, 3, 1, 5])
    assert_raises(ValueError, pls_array, c, 6, 0, 5, 0)
    
    # Case 5: length < min_length. Raises ValueError
    d = np.array([2, 4])
    assert_raises(ValueError, pls_array, d, 6, 3, 5, 0)
    
    # Case 6: length > max_length. Raises ValueError
    e = np.array([2, 4, 0, 3, 1, 5, 4, 3, 2])
    assert_raises(ValueError, pls_array, e, 6, 3, 5, 0)
    
    
def test_pls_blocks():
    a = np.array([0, 3, 5, 8, 9, 15])
    b = np.array([3, 5, 8, 9, 15])
    c = np.array([0, 3, 5, 8, 9])
    
    true_block = np.array([3, 5, 8, 9])
    
    # Check results
    block_orig = pls_blocks(true_block, max_entry=15)
    block_a = pls_blocks(a, max_entry=15)
    block_b = pls_blocks(b, max_entry=15)
    block_c = pls_blocks(c, max_entry=15)
    
    assert_array_equal(block_orig, true_block)    # No change
    assert_array_equal(block_a, true_block)
    assert_array_equal(block_b, true_block)    
    assert_array_equal(block_c, true_block)

    # Check that UserWarning is raised
    assert_warns(UserWarning, pls_blocks, a, 15, 0, True)
    assert_warns(UserWarning, pls_blocks, b, 15, 0, True)
    assert_warns(UserWarning, pls_blocks, c, 15, 0, True)
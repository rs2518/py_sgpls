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
# import pytest
import os

from sgpls import _pls as pls_
from sgpls import _spls as spls_
from sgpls import _gpls as gpls_
from sgpls import _sgpls as sgpls_

from sklearn.cross_decomposition import _pls as sklearn_pls_

import pandas as pd
import numpy as np


# =============================================================================
# NOTES
# =============================================================================
# There are Python packages that can run R code and handle R objects
# however, neither worked in this case (-_-):
#    - pyreadr parse method does not work on the models that were created 
#    in R containing the information from the sgPLS implementation.
#    
#    - rpy2 cannot be imported because it requires an older version of Python.


data_path = "Desktop/py_sgpls/data/"
# folders = [f for f in os.listdir(path) if f.endswith("pls")]


def load_csv_data(directory):
    """Returns dictionary of data obtained from a directory
    
    Collects data loaded from all '.csv' files found in the given directory
    into a dictionary. Data in the dictionary can be accessed using the
    corresponding file name without the '.csv' extension.
    """
    dict_df = {}
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, file), index_col=0)
            # Drop strange column name
            if len(df.columns)==1 and list(df.columns)==['get(model)[[item]]']:
                df.rename(columns={'get(model)[[item]]': ''}, inplace=True)
            dict_df[file[0:file.find('.csv')]] = df
    
    return dict_df


sPLS_Rmodel = load_csv_data(os.path.join(data_path, 'sgPLS_spls'))
gPLS_Rmodel = load_csv_data(os.path.join(data_path, 'sgPLS_gpls'))
sgPLS_Rmodel = load_csv_data(os.path.join(data_path, 'sgPLS_sgpls'))



# X_matrix = pd.read_csv(directory+'/data/simulated_data.csv',
#                        dtype='float').drop(columns=['Unnamed: 0'])
# 
# Y_matrix = pd.read_csv(directory+'/data/simulated_target.csv',
#                        dtype='float').drop(columns=['Unnamed: 0'])
# 
# X = X_matrix.values
# Y = Y_matrix.values

X = sPLS_Rmodel['X'].values
Y = sPLS_Rmodel['Y'].values

n_components = 2

x_vars = y_vars = np.array([60, 60])

x_groups = y_groups = np.array([4, 4])
x_block = np.arange(20, 400, 20)
y_block = np.arange(20, 500, 20)

alpha_x = alpha_y = np.array([0.95, 0.95])

# =============================================================================
# vs. sklearn - PLS
# =============================================================================

# Testing the PLS algorithm against sklearn alone should suffice since
# sklearn's has been thoroughly vetted and the py_sgpls implementation is
# taken from sklearn.

# Regression mode
plsreg = pls_.PLSRegression(n_components=n_components)
plsreg.fit(X, Y)
plsreg_sklearn = sklearn_pls_.PLSRegression(n_components=n_components)
plsreg_sklearn.fit(X, Y)


np.testing.assert_array_almost_equal(plsreg.x_scores_,
                                     plsreg_sklearn.x_scores_, decimal=6)
np.testing.assert_array_almost_equal(plsreg.x_loadings_,
                                     plsreg_sklearn.x_loadings_, decimal=6)
np.testing.assert_array_almost_equal(plsreg.x_weights_,
                                     plsreg_sklearn.x_weights_, decimal=6)
np.testing.assert_array_almost_equal(plsreg.y_scores_,
                                     plsreg_sklearn.y_scores_, decimal=6)
np.testing.assert_array_almost_equal(plsreg.y_loadings_,
                                     plsreg_sklearn.y_loadings_, decimal=6)
np.testing.assert_array_almost_equal(plsreg.y_weights_,
                                     plsreg_sklearn.y_weights_, decimal=6)


# Canonical Mode
plsca = pls_.PLSCanonical(n_components=n_components)
plsca.fit(X, Y)
plsca_sklearn = sklearn_pls_.PLSCanonical(n_components=n_components)
plsca_sklearn.fit(X, Y)


np.testing.assert_array_almost_equal(plsca.x_scores_,
                                     plsca_sklearn.x_scores_, decimal=6)
np.testing.assert_array_almost_equal(plsca.x_loadings_,
                                     plsca_sklearn.x_loadings_, decimal=6)
np.testing.assert_array_almost_equal(plsca.x_weights_,
                                     plsca_sklearn.x_weights_, decimal=6)
np.testing.assert_array_almost_equal(plsca.y_scores_,
                                     plsca_sklearn.y_scores_, decimal=6)
np.testing.assert_array_almost_equal(plsca.y_loadings_,
                                     plsca_sklearn.y_loadings_, decimal=6)
np.testing.assert_array_almost_equal(plsca.y_weights_,
                                     plsca_sklearn.y_weights_, decimal=6)

# ***** SUCCESS! PLS models seem to work

# =============================================================================
# vs. R package sgPLS - sPLS
# =============================================================================
# Regression mode
splsreg = spls_.sPLSRegression(x_vars=x_vars, y_vars=y_vars,
                               n_components=n_components)
splsreg.fit(X, Y)


np.testing.assert_array_almost_equal(splsreg.x_scores_,
                                     sPLS_Rmodel['variates_1'], decimal=6)
np.testing.assert_array_almost_equal(splsreg.x_loadings_,
                                     sPLS_Rmodel['mat.c'], decimal=6)
np.testing.assert_array_almost_equal(splsreg.x_weights_,
                                     sPLS_Rmodel['loadings_1'], decimal=6)
np.testing.assert_array_almost_equal(splsreg.y_scores_,
                                     sPLS_Rmodel['variates_2'], decimal=6)
np.testing.assert_array_almost_equal(splsreg.y_loadings_,
                                     sPLS_Rmodel['loadings_2'], decimal=6)
np.testing.assert_array_almost_equal(splsreg.y_weights_,
                                     sPLS_Rmodel['mat.d'], decimal=6)

# ***** ERROR! Small errors appearing. Check 'eps' from inner loops 

# NOTE: Run models for regression and canonical modes. Consider renaming
# R models to avoid confusion between modes


# # Canonical Mode
# splsca = spls_.sPLSCanonical(n_components=n_components,
#                               x_vars=x_vars, y_vars=y_vars)
# splsca.fit(X, Y)
# 
# 
# np.testing.assert_array_almost_equal(splsca.x_scores_,
#                                      sPLS_Rmodel['variates_1'], decimal=6)
# np.testing.assert_array_almost_equal(splsca.x_loadings_,
#                                      sPLS_Rmodel['mat.c'], decimal=6)
# np.testing.assert_array_almost_equal(splsca.x_weights_,
#                                      sPLS_Rmodel['loadings_1'], decimal=6)
# np.testing.assert_array_almost_equal(splsca.y_scores_,
#                                      sPLS_Rmodel['variates_2'], decimal=6)
# np.testing.assert_array_almost_equal(splsca.y_loadings_,
#                                      sPLS_Rmodel['loadings_2'], decimal=6)
# np.testing.assert_array_almost_equal(splsca.y_weights_,
#                                      sPLS_Rmodel['mat.e'], decimal=6)




# Quick check if gPLS/sgPLS run
gplsreg = gpls_.gPLSRegression(x_groups=x_groups, y_groups=y_groups,
                               x_block=x_block, y_block=y_block,
                               n_components=n_components)
gplsreg.fit(X, Y)



sgplsreg = sgpls_.sgPLSRegression(x_groups=x_groups, y_groups=y_groups,
                                  x_block=x_block, y_block=y_block,
                                  alpha_x=alpha_x, alpha_y=alpha_y,
                                  n_components=n_components)
sgplsreg.fit(X, Y)


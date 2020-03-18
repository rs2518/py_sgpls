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
from sklearn.cross_decomposition import _pls as skpls_

import pandas as pd
import numpy as np



# NOTE: There are Python packages that can run R code and handle R objects
# however, neither worked in this case (-_-):
#    - pyreadr parse method does not work on the models that were created 
#    in R containing the information from the sgPLS implementation.
#    
#    - rpy2 cannot be imported because it requires an older version of Python.

path = "Desktop/py_sgpls/data/"
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


sPLS_Rmodel = load_csv_data(os.path.join(path, 'sgPLS_spls'))
gPLS_Rmodel = load_csv_data(os.path.join(path, 'sgPLS_gpls'))
sgPLS_Rmodel = load_csv_data(os.path.join(path, 'sgPLS_sgpls'))



# X_matrix = pd.read_csv(directory+'/data/simulated_data.csv',
#                        dtype='float').drop(columns=['Unnamed: 0'])

# Y_matrix = pd.read_csv(directory+'/data/simulated_target.csv',
#                        dtype='float').drop(columns=['Unnamed: 0'])

# X = X_matrix.values
# Y = Y_matrix.values


n_components = 3

# =============================================================================
# Regression Mode
# =============================================================================
# Train sgpls and sklearn models
pls_sgpls = pls_.PLSRegression(n_components=n_components)
pls_sgpls.fit(X, Y)
pls_sklearn = skpls_.PLSRegression(n_components=n_components)
pls_sklearn.fit(X, Y)

# sgpls attributes
T = pls_sgpls.x_scores_
P = pls_sgpls.x_loadings_
W = pls_sgpls.x_weights_
U = pls_sgpls.y_scores_
Q = pls_sgpls.y_loadings_
C = pls_sgpls.y_weights_
# sklearn attributes
Ts = pls_sklearn.x_scores_
Ps = pls_sklearn.x_loadings_
Ws = pls_sklearn.x_weights_
Us = pls_sklearn.y_scores_
Qs = pls_sklearn.y_loadings_
Cs = pls_sklearn.y_weights_

# Check that pls attributes match sklearn
np.testing.assert_array_almost_equal(T, Ts, decimal=5)
np.testing.assert_array_almost_equal(P, Ps, decimal=5)
np.testing.assert_array_almost_equal(W, Ws, decimal=5)
np.testing.assert_array_almost_equal(U, Us, decimal=5)
np.testing.assert_array_almost_equal(Q, Qs, decimal=5)
np.testing.assert_array_almost_equal(C, Cs, decimal=5)
# All seem to match


# =============================================================================
# Canonical Mode
# =============================================================================
# Train sgpls and sklearn models
pls_sgpls = pls_.PLSCanonical(n_components=n_components)
pls_sgpls.fit(X, Y)
pls_sklearn = skpls_.PLSCanonical(n_components=n_components)
pls_sklearn.fit(X, Y)

# sgpls attributes
T = pls_sgpls.x_scores_
P = pls_sgpls.x_loadings_
W = pls_sgpls.x_weights_
U = pls_sgpls.y_scores_
Q = pls_sgpls.y_loadings_
C = pls_sgpls.y_weights_
# sklearn attributes
Ts = pls_sklearn.x_scores_
Ps = pls_sklearn.x_loadings_
Ws = pls_sklearn.x_weights_
Us = pls_sklearn.y_scores_
Qs = pls_sklearn.y_loadings_
Cs = pls_sklearn.y_weights_

# Check that pls attributes match sklearn
np.testing.assert_array_almost_equal(T, Ts, decimal=5)
np.testing.assert_array_almost_equal(P, Ps, decimal=5)
np.testing.assert_array_almost_equal(W, Ws, decimal=5)
np.testing.assert_array_almost_equal(U, Us, decimal=5)
np.testing.assert_array_almost_equal(Q, Qs, decimal=5)
np.testing.assert_array_almost_equal(C, Cs, decimal=5)
# All seem to match

# =============================================================================
# 
# =============================================================================

# =============================================================================
# TEMPORARY IMPORT. REMOVE WHEN FINALISED
# =============================================================================
# Import locally
import os, sys

directory = 'Desktop/py_sgpls'
path = os.path.join(os.path.abspath('.'), directory)
sys.path.append(path)
# =============================================================================
import os

from sgpls import _pls as pls_
from sgpls import _spls as spls_
from sgpls import _gpls as gpls_
from sgpls import _sgpls as sgpls_
from sgpls import _plsda as plsda_
# from sgpls import _splsda as splsda_
# from sgpls import _gplsda as gplsda_
# from sgpls import _sgplsda as sgplsda_

from sklearn.cross_decomposition import _pls as sklearn_pls_
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np


# NOTES
# ----------------------------------------------------------------------------
# There are Python packages that can run R code and handle R objects
# however, neither worked in this case (-_-):
#    - pyreadr parse method does not work on the models that were created 
#    in R containing the information from the sgPLS implementation.
#    
#    - rpy2 cannot be imported because it requires an older version of Python.


# Load R data into dictionary
def load_csv_data(directory, find=None):
    """Returns dictionary of data obtained from a directory
    
    Collects data loaded from all '.csv' files found in the given directory
    into a dictionary. Search can be narrowed to files which contain
    specific characters given by 'find'.
    Data in the dictionary can be accessed using the corresponding file
    name without the '.csv' extension.
    """
    if find is None:
        find = ""
        
    dict_df = {}
    for file in os.listdir(directory):
        if file.endswith('.csv') and find in file:
            df = pd.read_csv(os.path.join(directory, file), index_col=0)
            # Drop strange column name
            if len(df.columns)==1 and list(df.columns)==['get(model)[[item]]']:
                df.rename(columns={'get(model)[[item]]': ''}, inplace=True)
            dict_df[file[0:file.find('.csv')]] = df
    
    return dict_df


# =============================================================================
# Regression problem - dataset 1
# =============================================================================


data_path = "Desktop/py_sgpls/data/dataset1"
# folders = [f for f in os.listdir(data_path) if f.endswith("pls")]

sPLS_R_reg = load_csv_data(os.path.join(data_path, 'sgPLS_spls'),
                           find="regression")
gPLS_R_reg = load_csv_data(os.path.join(data_path, 'sgPLS_gpls'),
                           find="regression")
sgPLS_R_reg = load_csv_data(os.path.join(data_path, 'sgPLS_sgpls'),
                            find="regression")
sPLS_R_ca = load_csv_data(os.path.join(data_path, 'sgPLS_spls'),
                          find="canonical")
gPLS_R_ca = load_csv_data(os.path.join(data_path, 'sgPLS_gpls'),
                          find="canonical")
sgPLS_R_ca = load_csv_data(os.path.join(data_path, 'sgPLS_sgpls'),
                           find="canonical")


X = pd.read_csv(os.path.join(data_path, "X.csv"), index_col = 0,
                dtype='float').values
Y = pd.read_csv(os.path.join(data_path, "Y.csv"), index_col = 0,
                dtype='float').values

n_components = 2
x_vars = y_vars = np.array([60, 60])
x_groups = y_groups = np.array([4, 4])
x_block = np.arange(20, 400, 20)
y_block = np.arange(20, 500, 20)
alpha_x = alpha_y = np.array([0.95, 0.95])


# Compare model results
# =====================

# vs. sklearn - PLS
# -----------------
# Testing the PLS algorithm against sklearn alone should suffice since
# sklearn's has been thoroughly vetted and the py_sgpls implementation is
# based upon sklearn's framework.

# Regression mode
# ---------------
plsreg = pls_.PLSRegression(n_components=n_components)
plsreg.fit(X, Y)
plsreg_sklearn = sklearn_pls_.PLSRegression(n_components=n_components)
plsreg_sklearn.fit(X, Y)

np.testing.assert_array_almost_equal(plsreg.x_scores_,
                                     plsreg_sklearn.x_scores_)
np.testing.assert_array_almost_equal(plsreg.x_loadings_,
                                     plsreg_sklearn.x_loadings_)
np.testing.assert_array_almost_equal(plsreg.x_weights_,
                                     plsreg_sklearn.x_weights_)
np.testing.assert_array_almost_equal(plsreg.y_scores_,
                                     plsreg_sklearn.y_scores_)
np.testing.assert_array_almost_equal(plsreg.y_loadings_,
                                     plsreg_sklearn.y_loadings_)
np.testing.assert_array_almost_equal(plsreg.y_weights_,
                                     plsreg_sklearn.y_weights_)

# Canonical Mode
# --------------
plsca = pls_.PLSCanonical(n_components=n_components)
plsca.fit(X, Y)
plsca_sklearn = sklearn_pls_.PLSCanonical(n_components=n_components)
plsca_sklearn.fit(X, Y)

np.testing.assert_array_almost_equal(plsca.x_scores_,
                                     plsca_sklearn.x_scores_)
np.testing.assert_array_almost_equal(plsca.x_loadings_,
                                     plsca_sklearn.x_loadings_)
np.testing.assert_array_almost_equal(plsca.x_weights_,
                                     plsca_sklearn.x_weights_)
np.testing.assert_array_almost_equal(plsca.y_scores_,
                                     plsca_sklearn.y_scores_)
np.testing.assert_array_almost_equal(plsca.y_loadings_,
                                     plsca_sklearn.y_loadings_)
np.testing.assert_array_almost_equal(plsca.y_weights_,
                                     plsca_sklearn.y_weights_)


# vs. R package sgPLS - sPLS
# --------------------------

# Regression mode
# ---------------
splsreg = spls_.sPLSRegression(x_vars=x_vars, y_vars=y_vars,
                               n_components=n_components)
splsreg.fit(X, Y)

np.testing.assert_array_almost_equal(splsreg.x_scores_,
                                     sPLS_R_reg['regression_variates_1'])
np.testing.assert_array_almost_equal(splsreg.x_loadings_,
                                     sPLS_R_reg['regression_mat.c'])
np.testing.assert_array_almost_equal(splsreg.x_weights_,
                                     sPLS_R_reg['regression_loadings_1'])
np.testing.assert_array_almost_equal(splsreg.y_scores_,
                                     sPLS_R_reg['regression_variates_2'])
np.testing.assert_array_almost_equal(splsreg.y_loadings_,
                                     sPLS_R_reg['regression_mat.d'])
np.testing.assert_array_almost_equal(splsreg.y_weights_,
                                     sPLS_R_reg['regression_loadings_2'])

# Canonical Mode
# --------------
splsca = spls_.sPLSCanonical(x_vars=x_vars, y_vars=y_vars,
                             n_components=n_components)
splsca.fit(X, Y)

np.testing.assert_array_almost_equal(splsca.x_scores_,
                                     sPLS_R_ca['canonical_variates_1'])
np.testing.assert_array_almost_equal(splsca.x_loadings_,
                                     sPLS_R_ca['canonical_mat.c'])
np.testing.assert_array_almost_equal(splsca.x_weights_,
                                     sPLS_R_ca['canonical_loadings_1'])
np.testing.assert_array_almost_equal(splsca.y_scores_,
                                     sPLS_R_ca['canonical_variates_2'])
np.testing.assert_array_almost_equal(splsca.y_loadings_,
                                     sPLS_R_ca['canonical_mat.e'])
np.testing.assert_array_almost_equal(splsca.y_weights_,
                                     sPLS_R_ca['canonical_loadings_2'])


# vs. R package sgPLS - gPLS
# --------------------------

# Regression mode
# ---------------
gplsreg = gpls_.gPLSRegression(x_groups=x_groups, y_groups=y_groups,
                               x_block=x_block, y_block=y_block,
                               n_components=n_components)
gplsreg.fit(X, Y)

np.testing.assert_array_almost_equal(gplsreg.x_scores_,
                                     gPLS_R_reg['regression_variates_1'])
np.testing.assert_array_almost_equal(gplsreg.x_loadings_,
                                     gPLS_R_reg['regression_mat.c'])
np.testing.assert_array_almost_equal(gplsreg.x_weights_,
                                     gPLS_R_reg['regression_loadings_1'])
np.testing.assert_array_almost_equal(gplsreg.y_scores_,
                                     gPLS_R_reg['regression_variates_2'])
np.testing.assert_array_almost_equal(gplsreg.y_loadings_,
                                     gPLS_R_reg['regression_mat.d'])
np.testing.assert_array_almost_equal(gplsreg.y_weights_,
                                     gPLS_R_reg['regression_loadings_2'])

# Canonical Mode
# --------------
gplsca = gpls_.gPLSCanonical(x_groups=x_groups, y_groups=y_groups,
                             x_block=x_block, y_block=y_block,
                             n_components=n_components)
gplsca.fit(X, Y)

np.testing.assert_array_almost_equal(gplsca.x_scores_,
                                     gPLS_R_ca['canonical_variates_1'])
np.testing.assert_array_almost_equal(gplsca.x_loadings_,
                                     gPLS_R_ca['canonical_mat.c'])
np.testing.assert_array_almost_equal(gplsca.x_weights_,
                                     gPLS_R_ca['canonical_loadings_1'])
np.testing.assert_array_almost_equal(gplsca.y_scores_,
                                     gPLS_R_ca['canonical_variates_2'])
np.testing.assert_array_almost_equal(gplsca.y_loadings_,
                                     gPLS_R_ca['canonical_mat.e'])
np.testing.assert_array_almost_equal(gplsca.y_weights_,
                                     gPLS_R_ca['canonical_loadings_2'])


# vs. R package sgPLS - sgPLS
# ---------------------------
# NOTE: Numerical errors will occur in the numerical methods used in the
# sgPLS algorithm. To ensure that the R and Python implementations agree up to
# a sufficiently degree, the R code has been temporarily overwritten so that
# its internal functions achieve great enough accuracy to be comparable to
# Python (which appears to achieve higher accuracy than R).
#
# Also, an inconsistency appears between the literature and original R code
# which affects the _sparse_group_thresholding function. Both versions have
# been added for comparison

# Regression mode
# ---------------
sgplsreg = sgpls_.sgPLSRegression(x_groups=x_groups, y_groups=y_groups,
                                  x_block=x_block, y_block=y_block,
                                  alpha_x=alpha_x, alpha_y=alpha_y,
                                  n_components=n_components)
sgplsreg.fit(X, Y)

np.testing.assert_array_almost_equal(sgplsreg.x_scores_,
                                     sgPLS_R_reg['regression_variates_1'])
np.testing.assert_array_almost_equal(sgplsreg.x_loadings_,
                                     sgPLS_R_reg['regression_mat.c'])
np.testing.assert_array_almost_equal(sgplsreg.x_weights_,
                                     sgPLS_R_reg['regression_loadings_1'])
np.testing.assert_array_almost_equal(sgplsreg.y_scores_,
                                     sgPLS_R_reg['regression_variates_2'])
np.testing.assert_array_almost_equal(sgplsreg.y_loadings_,
                                     sgPLS_R_reg['regression_mat.d'])
np.testing.assert_array_almost_equal(sgplsreg.y_weights_,
                                     sgPLS_R_reg['regression_loadings_2'])

# Canonical Mode
# --------------
sgplsca = sgpls_.sgPLSCanonical(x_groups=x_groups, y_groups=y_groups,
                                x_block=x_block, y_block=y_block,
                                alpha_x=alpha_x, alpha_y=alpha_y,
                                n_components=n_components)
sgplsca.fit(X, Y)

np.testing.assert_array_almost_equal(sgplsca.x_scores_,
                                     sgPLS_R_ca['canonical_variates_1'])
np.testing.assert_array_almost_equal(sgplsca.x_loadings_,
                                     sgPLS_R_ca['canonical_mat.c'])
np.testing.assert_array_almost_equal(sgplsca.x_weights_,
                                     sgPLS_R_ca['canonical_loadings_1'])
np.testing.assert_array_almost_equal(sgplsca.y_scores_,
                                     sgPLS_R_ca['canonical_variates_2'])
np.testing.assert_array_almost_equal(sgplsca.y_loadings_,
                                     sgPLS_R_ca['canonical_mat.e'])
np.testing.assert_array_almost_equal(sgplsca.y_weights_,
                                     sgPLS_R_ca['canonical_loadings_2'])



# =============================================================================
# Classification problem - dataset 2
# =============================================================================


data_path2 = "Desktop/py_sgpls/data/dataset2"


PLSDA_R = load_csv_data(os.path.join(data_path2, 'sgPLS_plsda'),
                           find="regression")
# sPLSDA_R = load_csv_data(os.path.join(data_path2, 'sgPLS_splsda'),
#                             find="regression")
gPLSDA_R = load_csv_data(os.path.join(data_path2, 'sgPLS_gplsda'),
                           find="regression")
sgPLSDA_R = load_csv_data(os.path.join(data_path2, 'sgPLS_sgplsda'),
                            find="regression")


df = pd.read_csv(os.path.join(data_path2, "simuData.csv"), index_col = 0)
X = df.iloc[:,:-1].values
le = LabelEncoder()
Y = le.fit_transform(df.iloc[:,-1].astype('category')).astype(float)

n_components = 3
# x_vars = np.array([60, 60])
x_groups = np.array([2, 2, 2])
x_block = np.insert(np.arange(100, 1000, 100), 2, 250)
alpha_x = np.array([0.5, 0.5, 0.99])


# Compare model results
# =====================

# vs. R package sgPLS - PLS-DA
# ----------------------------
# Although sgPLS does not explicitly have a PLS-DA implementation, we can run
# PLS-DA but running sPLS-DA with no sparsity

plsda = plsda_.PLSDARegression(n_components=n_components)
plsda.fit(X, Y)

np.testing.assert_array_almost_equal(plsda.x_scores_,
                                     PLSDA_R['regression_variates_1'])
np.testing.assert_array_almost_equal(plsda.x_loadings_,
                                     PLSDA_R['regression_mat.c'])
np.testing.assert_array_almost_equal(plsda.x_weights_,
                                     PLSDA_R['regression_loadings_1'])
np.testing.assert_array_almost_equal(plsda.y_scores_,
                                     PLSDA_R['regression_variates_2'])
np.testing.assert_array_almost_equal(plsda.y_loadings_,
                                     PLSDA_R['regression_mat.d'])
np.testing.assert_array_almost_equal(plsda.y_weights_,
                                     PLSDA_R['regression_loadings_2'])
# Values show small errors and sign differences because of numerical errors
# from the approximation given by NIPALS compared to SVD results
import os

import pandas as pd
import numpy as np

# from sklearn.cross_decomposition import _pls as sklearn_pls_
# from sklearn.preprocessing import LabelEncoder

# from sgpls import _pls as pls_
from sgpls import _spls as spls_
# from sgpls import _gpls as gpls_
# from sgpls import _sgpls as sgpls_
# from sgpls import _plsda as plsda_
# from sgpls import _splsda as splsda_
# from sgpls import _gplsda as gplsda_
# from sgpls import _sgplsda as sgplsda_



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


data_path = "Desktop/Git/py_sgpls/data/dataset1"
# folders = [f for f in os.listdir(data_path) if f.endswith("pls")]

sPLS_R_reg = load_csv_data(os.path.join(data_path, 'sgPLS_spls'),
                           find="regression")
sPLS_R_ca = load_csv_data(os.path.join(data_path, 'sgPLS_spls'),
                          find="canonical")


X = pd.read_csv(os.path.join(data_path, "X.csv"), index_col = 0,
                dtype='float').values
Y = pd.read_csv(os.path.join(data_path, "Y.csv"), index_col = 0,
                dtype='float').values

n_components = 2
x_vars = y_vars = np.array([60, 60])
# x_groups = y_groups = np.array([4, 4])
# x_block = np.arange(20, 400, 20)
# y_block = np.arange(20, 500, 20)
# alpha_x = alpha_y = np.array([0.95, 0.95])


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

# Both regression and canonical match when norm_y_weight = True
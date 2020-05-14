from ._pls import PLSCanonical, PLSRegression
from ._spls import sPLSCanonical, sPLSRegression
from ._gpls import gPLSCanonical, gPLSRegression
from ._sgpls import sgPLSCanonical, sgPLSRegression
from ._plsda import PLSDACanonical, PLSDARegression
from ._splsda import sPLSDACanonical, sPLSDARegression
from ._gplsda import gPLSDACanonical, gPLSDARegression
from ._sgplsda import sgPLSDACanonical, sgPLSDARegression

__all__ = ['PLSCanonical', 'PLSRegression',
           'sPLSCanonical', 'sPLSRegression',
           'gPLSCanonical', 'gPLSRegression',
           'sgPLSCanonical', 'sgPLSRegression',
           'PLSDACanonical', 'PLSDARegression',
           'sPLSDACanonical', 'sPLSDARegression',
           'gPLSDACanonical', 'gPLSDARegression',
           'sgPLSDACanonical', 'sgPLSDARegression']
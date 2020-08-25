from scipy import sparse
from ..libmesh.fypymesh import *
from typing import Union,Tuple

import scipy.sparse.linalg

TOUT = Tuple(sparse.coo_matrix,sparse.coo_matrix)
def assembly(fypymesh:FyPyMesh)->TOUT:
    pass

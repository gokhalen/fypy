from scipy import sparse
from ..libmesh.fypymesh import *
from typing import Union,Tuple

import scipy.sparse.linalg

ss   = sparse.coo_matrix
TOUT = Tuple[ss,ss]

def build_k_matrix(fypymesh:FyPyMesh)->ss:
    pass

def build_rhs(fypymesh:FyPyMesh)->ss:
    pass

def assembly(fypymesh:FyPyMesh)->TOUT:
    kk  = build_k_matrix(fypymesh)
    rhs = build_rhs(fypymesh)
    return (kk,rhs)

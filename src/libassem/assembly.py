from scipy import sparse
import scipy.sparse.linalg
import numpy as np

from ..libmesh.fypymesh import *

# get elements
from ..libelem.elembase import *
from ..libelem.linelas1d import *


from typing import Union,Tuple


ss         = sparse.coo_matrix
TOUTASS    = Tuple[ss,ss]
TOUTGETELM = Union[LinElas1D]

def getelem(elemname:str,ninteg,gdofn)->TOUTGETELM:
    if ( elemname == 'linelas1d'):
        return LinElas1D(ninteg=ninteg,gdofn=gdofn)
    

def assembly(fypymesh:FyPyMesh)->TOUTASS:
    gdofn  = fypymesh.gdofn
    ninteg = fypymesh.ninteg
    
    # initialize zero global matrix and global rhs vector 
    data = (0.0,); row = (0,); col = (0,); tt = (data,(row,col))
    kk  = sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64');
    rhs = sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64');
    
    for ielem in range(1,fypymesh.nelem+1):
        eltype = fypymesh.conn[ielem-1][-1]
        # nn is nodes 
        *nn,   =  fypymesh.conn[ielem-1][0:-1]
        elem   = getelem(eltype,ninteg,gdofn)
        coord  = [np.asarray(fypymesh.coord[n-1]) for n in nn]
        coord  = np.asarray(coord)

        # call the setdata method to initialize the element
        elem.compute()
    
    return (kk,rhs)

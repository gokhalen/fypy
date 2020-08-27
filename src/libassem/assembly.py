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

    raise RuntimeError(f'element {elemname} not found')
    

def assembly(fypymesh:FyPyMesh)->TOUTASS:
    gdofn  = fypymesh.gdofn
    ninteg = fypymesh.ninteg
    
    # initialize zero global matrix and global rhs vector 
    data = (0.0,); row = (0,); col = (0,); tt = (data,(row,col))
    kk  = sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64');
    rhs = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64');
    
    for ielem in range(1,fypymesh.nelem+1):
        eltype = fypymesh.conn[ielem-1][-1]
        elem   = getelem(eltype,ninteg,gdofn)
        
        # nn is list of nodes 
        *nn,   =  fypymesh.conn[ielem-1][0:-1]

        # get coord
        coord  = np.asarray([ fypymesh.coord[n-1]  for n in nn])
        prop   = np.asarray([ fypymesh.prop[n-1]   for n in nn])
        bf     = np.asarray([ fypymesh.bf[n-1]     for n in nn])
        pforce = np.asarray([ fypymesh.pforce[n-1] for n in nn])
        dirich = np.asarray([ fypymesh.dirich[n-1] for n in nn])
        trac   = np.asarray([ fypymesh.trac[n-1]   for n in nn])
        ideqn  = np.asarray([ fypymesh.ideqn[n-1]  for n in nn])

        # call the setdata method to initialize the element
        elem.setdata(coord=coord,prop=prop,bf=bf,pforce=pforce,dirich=dirich,trac=trac,ideqn=ideqn)

        # compute
        elem.compute()

        kk  += elem.kmatrix
        rhs += elem.rhs
    
        
    return (kk,rhs)

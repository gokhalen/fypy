from scipy import sparse
import scipy.sparse.linalg
import numpy as np
import time

from ..libmesh import *

from .assutils import *

from typing import Union,Tuple


ss         = sparse.coo_matrix
TOUTASS    = Tuple[ss,ss]
TOUTGETELM = Union[LinElas1D]


def mapelem(tt):
    elem = getelem(tt.elype,tt.ninteg,tt.gdofn)
    pass

def assembly_par(fypymesh:FyPyMesh,nprocs:int)->TOUTASS:
    gdofn  = fypymesh.gdofn
    ninteg = fypymesh.ninteg
    
    # initialize zero global matrix and global rhs vector 
    data = (0.0,); row = (0,); col = (0,); tt = (data,(row,col))
    kk  = sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64');
    rhs = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64');

    eldict = {}

    scipy_time = 0.0

    itr_elem = iter(fypymesh)
     
    for ielem1, tt in enumerate(itr_elem):
        ielem  = ielem1+1
        eltype = tt.eltype
        elem   = getelem(eltype,ninteg,gdofn)
        nn     = tt.nodes
        coord  = tt.coord
        prop   = tt.prop
        bf     = tt.bf
        pforce = tt.pforce
        dirich = tt.dirich
        trac   = tt.trac
        ideqn  = tt.ideqn
        elem.setdata(coord=coord,prop=prop,bf=bf,pforce=pforce,dirich=dirich,trac=trac,ideqn=ideqn)
        # compute
        elem.compute()

        t1   = time.perf_counter()
        kk  += elem.kmatrix
        rhs += elem.rhs
        t2   = time.perf_counter()
        scipy_time += (t2-t1)
        
    
        
    return (kk,rhs,scipy_time)

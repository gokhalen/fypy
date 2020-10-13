import scipy.sparse.linalg
import numpy as np
import time
import multiprocessing as mp

from ..libmesh import *
from scipy import sparse
from .assutils import *
from typing import Union,Tuple


ss         = sparse.coo_matrix
TOUTASS    = Tuple[ss,ss]
TOUTGETELM = Union[LinElas1D]

class KKRhs():
    '''
     this class holds the element stiffness matrices and right hand size
     since the elements return sparse matrices and rhs vectors
     which have dimensions of total degrees of freedom, they're effectively 
     global matrices and rhs vectors which can be added together 
     to get the 'assembled' matrix and rhs
    
     conceptually, we want to drive the assembly step by the following lines of code
 
     kk  = sum(mapelem,iter(fypymesh))
     rhs = sum(mapelem,iter(fypymesh))

     instead of iterating over the elements twice, and localizing data twice
     we can combine both steps into one

     kkrhs = sum(mapelem,iter(fypymesh))

     this requires an object which holds kk and rhs and supports the '+' operator
     that object is this KKRhs
 
    '''
    def __init__(self,gdofn):
        # initialize to zero vector and matrix of dimension gdofn
        self.gdofn = gdofn
        data = (0.0,); row = (0,); col = (0,); tt = (data,(row,col))
        self.kk    = scipy.sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64');
        self.rhs   = scipy.sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64');

    def __add__(self,obj):
        out = KKRhs(self.gdofn)
        out.kk  = self.kk  + obj.kk
        out.rhs = self.rhs + obj.rhs
        return out

def mapelem(tt):
    elem = getelem(tt.eltype,tt.ninteg,tt.gdofn)
    elem.setdata(coord=tt.coord  , prop=tt.prop, bf=tt.bf,      pforce=tt.pforce,
                 dirich=tt.dirich, trac=tt.trac, ideqn=tt.ideqn                 )
    elem.compute()
    kkrhs_e     = KKRhs(tt.gdofn)
    kkrhs_e.kk  = elem.kmatrix
    kkrhs_e.rhs = elem.rhs 
    # return (elem.kmatrix,elem.rhs)
    return kkrhs_e

def assembly_par(fypymesh:FyPyMesh,nprocs:int,chunksize:int)->TOUTASS:
    gdofn  = fypymesh.gdofn
    ninteg = fypymesh.ninteg
    
    # initialize zero global matrix and global rhs vector 
    data = (0.0,); row = (0,); col = (0,); tt = (data,(row,col))
    kk  = sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64');
    rhs = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64');

    scipy_time = 0.0

    itr_elem = iter(fypymesh)

    # global,assembled kkrhs, hence the g
    kkrhs_g    = KKRhs(fypymesh.gdofn)
    # initializer for sum
    kkrhs_zero = KKRhs(fypymesh.gdofn)

    with mp.Pool(processes=nprocs) as pool:
        kkrhs_g = sum(pool.imap(mapelem,itr_elem,chunksize=chunksize),start=kkrhs_zero)
    
    '''
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


        # kkmap,rhsmap=mapelem(tt)
        kkrhs_e     = mapelem(tt)
        t1          = time.perf_counter()
        kkrhs_g    += kkrhs_e
        t2          = time.perf_counter()
        scipy_time += (t2-t1)
    '''
    
    return (kkrhs_g.kk,kkrhs_g.rhs,scipy_time)

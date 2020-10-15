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


class KKRhsRaw():
    '''
    class on similar lines to KKRhs but it works with the raw data, rather than SciPy matrices
    '''
    def __init__(self):
        self.kdata = np.asarray([0])
        self.krow  = np.asarray([0])
        self.kcol  = np.asarray([0])
        
        self.rhsdata = np.asarray([0])
        self.rhsrow  = np.asarray([0])
        self.rhscol  = np.asarray([0])

    def __add__(self,obj):
        out = KKRhsRaw()
        out.kdata = np.concatenate([self.kdata,obj.kdata])
        out.krow  = np.concatenate([self.krow,obj.krow])
        out.kcol  = np.concatenate([self.kcol,obj.kcol])
        
        out.rhsdata = np.concatenate([self.rhsdata,obj.rhsdata])
        out.rhsrow  = np.concatenate([self.rhsrow,obj.rhsrow])
        out.rhscol  = np.concatenate([self.rhscol,obj.rhscol])
        
        return out

def mapelem(tt):
    elem = getelem(tt.eltype,tt.ninteg,tt.gdofn)
    elem.setdata(coord=tt.coord  , prop=tt.prop, bf=tt.bf,      pforce=tt.pforce,
                 dirich=tt.dirich, trac=tt.trac, ideqn=tt.ideqn                 )
    elem.compute()

    kkrhs_e  = KKRhsRaw()
    
    kkrhs_e.kdata = elem.kdata
    kkrhs_e.krow  = elem.krow
    kkrhs_e.kcol  = elem.kcol

    kkrhs_e.rhsdata = elem.fdata
    kkrhs_e.rhsrow  = elem.frow
    kkrhs_e.rhscol  = elem.fcol
    
    # kkrhs_e     = KKRhs(tt.gdofn)
    # kkrhs_e.kk  = elem.kmatrix
    # kkrhs_e.rhs = elem.rhs 
    # return (elem.kmatrix,elem.rhs)
    return kkrhs_e

def assembly_par(fypymesh:FyPyMesh,nprocs:int,chunksize:int)->TOUTASS:
    gdofn  = fypymesh.gdofn
    ninteg = fypymesh.ninteg
    
    # initialize zero global matrix and global rhs vector 
    # data = (0.0,); row = (0,); col = (0,); tt = (data,(row,col))
    # kk  = sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64');
    # rhs = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64');

    scipy_time = 0.0

    itr_elem = iter(FyPyMeshItr(fypymesh,0,fypymesh.nelem))

    # global,assembled kkrhs, hence the g
    # kkrhs_g    = KKRhs(fypymesh.gdofn)
    kkrhs_g    = KKRhsRaw()
    # initializer for sum
    # kkrhs_zero = KKRhs(fypymesh.gdofn)
    kkrhs_zero = KKRhsRaw()

    # imap with chunksize=1 seems to give best performance
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
    tt  = (kkrhs_g.kdata,(kkrhs_g.krow,kkrhs_g.kcol))
    kk  = sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64');
    
    tt  = (kkrhs_g.rhsdata,(kkrhs_g.rhsrow,kkrhs_g.rhscol))
    rhs = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64');
       
    return (kk,rhs,scipy_time)


def assembly_async(fypymesh:FyPyMesh,nprocs:int,chunksize:int)->TOUTASS:
    assert ( ( fypymesh.nelem % nprocs ) == 0 ),f'{fypymesh.nelem=} must be divisible by {nprocs=}'
    gdofn  = fypymesh.gdofn
    ninteg = fypymesh.ninteg
    nstep  = int(fypymesh.nelem / nprocs)
    
    # create iterators
    itrlist   = [ iter(FyPyMeshItr(fypymesh,i*(nstep),(i+1)*nstep)) for i in range(nprocs)]
    maplist   = [ map(mapelem,itrlist[i])                           for i in range(nprocs)]
    
    with mp.Pool(processes=nprocs) as pool:
        asynclist = [pool.apply_async(sum,(maplist[i],KKRhsRaw() )) for i in range(nprocs)]
        reslist   = [res.get(timeout=None) for res in asynclist]

    kkrhs_g=sum(reslist,KKRhsRaw())

    tt  = (kkrhs_g.kdata,(kkrhs_g.krow,kkrhs_g.kcol))
    kk  = sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64');
    
    tt  = (kkrhs_g.rhsdata,(kkrhs_g.rhsrow,kkrhs_g.rhscol))
    rhs = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64');

    scipy_time = 0.0
        
    return (kk,rhs,scipy_time)

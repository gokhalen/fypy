import scipy.sparse.linalg
import numpy as np
import multiprocessing as mp
from timerit import Timer

from ..libmesh import *
from scipy import sparse
from .assutils import *
from typing import Union,Tuple

# NOTE: The List Based Assembly (KKRhsRawList) is slightly slower for 4 processes
#       than KKRhsRaw using async assembly (2.2 secs vs 1.5 secs)
# I think there is loading time being wasted. This can possibly be cut down
# by initializing a FyPy object and using it repeatedly.

ss         = sparse.coo_matrix
TOUTASS    = Tuple[ss,ss]
TOUTGETELM = Union[LinElas1D]

def assembly_poolmap(fypymesh:FyPyMesh,nprocs:int,chunksize:int)->TOUTASS:
    gdofn  = fypymesh.gdofn
    ninteg = fypymesh.ninteg
    
    # initialize zero global matrix and global rhs vector 
    # data = (0.0,); row = (0,); col = (0,); tt = (data,(row,col))
    # kk  = sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64');
    # rhs = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64');

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

    treduc = Timer('FyPy Reduction Time',verbose=0)
    with treduc:
        tt  = (kkrhs_g.kdata,(kkrhs_g.krow,kkrhs_g.kcol))
        kk  = sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64');
    
        tt  = (kkrhs_g.rhsdata,(kkrhs_g.rhsrow,kkrhs_g.rhscol))
        rhs = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64');
       
    return (kk,rhs,treduc.elapsed)


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

    treduc = Timer('FyPy Reduction Time',verbose=0)
    with treduc:
        kkrhs_g=sum(reslist,KKRhsRaw())
        tt  = (kkrhs_g.kdata,(kkrhs_g.krow,kkrhs_g.kcol))
        kk  = sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64');

        tt  = (kkrhs_g.rhsdata,(kkrhs_g.rhsrow,kkrhs_g.rhscol))
        rhs = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64');
        
    return (kk,rhs,treduc.elapsed)

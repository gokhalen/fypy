import scipy.sparse.linalg
import numpy as np
import multiprocessing as mp
from timerit import Timer

from ..libmesh import *
from scipy import sparse
from .assutils import *
from typing import Union,Tuple

# NOTE: The List Based Assembly (KKRhsRawList) is slightly slower for 4 processes
#       than KKRhsRaw using async assembly (2.2 secs vs 1.5 secs) for 64x64 problem
#       a 128x128 problem is only 1.5 times faster using 4 procs (list assembly)
#       over a serial problem (list)
# I think there is loading time being wasted. This can possibly be cut down
# by initializing a FyPy object and using it repeatedly.

# speed up for Assembly is possibly slow, most likely, due to the sum(map()) paradigm
# I think it is the operator overloading that makes it slow
# better solution is to use list assembly on each process and return lists

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
    kkrhs_g    = KKRhsRawList()
    # initializer for sum
    # kkrhs_zero = KKRhs(fypymesh.gdofn)
    kkrhs_zero = KKRhsRawList()

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
        asynclist = [pool.apply_async(sum,(maplist[i],KKRhsRawList() )) for i in range(nprocs)]
        reslist   = [res.get(timeout=None) for res in asynclist]

    treduc = Timer('FyPy Reduction Time',verbose=0)
    with treduc:
        kkrhs_g=sum(reslist,KKRhsRawList())
        tt  = (kkrhs_g.kdata,(kkrhs_g.krow,kkrhs_g.kcol))
        kk  = sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64');

        tt  = (kkrhs_g.rhsdata,(kkrhs_g.rhsrow,kkrhs_g.rhscol))
        rhs = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64');
        
    return (kk,rhs,treduc.elapsed)


def assembly_parlist(fypymesh,nprocs,chunksize):
    gdofn  = fypymesh.gdofn
    ninteg = fypymesh.ninteg

    if ( fypymesh.nelem < nprocs):
        sys.exit(f'{fypymesh.nelem=} should not be less than {fypymesh.nprocs}')

    step = int(fypymesh.nelem / nprocs)
    startelemlist = []
    endelemlist   = []
    for istart in range(1,fypymesh.nelem+1,step):
        startelemlist.append(istart)
        endelemlist.append(istart+step)

    endelemlist[-1] = fypymesh.nelem+1

    print(startelemlist,endelemlist)

    
    karr   = [0]; krow   = [0]; kcol   = [0]
    rhsarr = [0]; rhsrow = [0]; rhscol = [0]


    with mp.Pool(processes=nprocs) as pool:
        asynclist = [pool.apply_async(parlistworker,(fypymesh,startelemlist[i],endelemlist[i])) for i in range(nprocs)]
        reslist   = [res.get(timeout=None) for res in asynclist]

    for tt in reslist:
        karr.extend(tt[0]);     krow.extend(tt[1]);     kcol.extend(tt[2])   
        rhsarr.extend(tt[3]);   rhsrow.extend(tt[4]);   rhscol.extend(tt[5]);
        
    #for ist,ien in zip(startelemlist,endelemlist):
    #    tt = parlistworker(fypymesh,ist,ien)
    #    
    #    karr.extend(tt[0]);     krow.extend(tt[1]);     kcol.extend(tt[2])
    #    rhsarr.extend(tt[3]);   rhsrow.extend(tt[4]);   rhscol.extend(tt[5]);

    kk   = sparse.coo_matrix((karr,(krow,kcol)),shape=(gdofn,gdofn),dtype='float64');
    rhs  = sparse.coo_matrix((rhsarr,(rhsrow,rhscol)),shape=(gdofn,1),dtype='float64');

    return (kk,rhs,0.0)

def parlistworker(fypymesh:FyPyMesh,start,end)->TOUTASS:
    # 1 based indexing. end is not included in range
    gdofn  = fypymesh.gdofn
    ninteg = fypymesh.ninteg
    
    # initialize zero global matrix and global rhs vector 
    data = (0.0,); row = (0,); col = (0,); tt = (data,(row,col))
    #kk  = sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64');
    #rhs = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64');

    karr   = []; krow   = []; kcol   = []
    rhsarr = []; rhsrow = []; rhscol = []

    # we refer to the process of combining element matrices and vectors as 'reduction'
    reduction_time = 0.0
    treduc         = Timer('FyPy Assembly (Reduction) Timer',verbose=0)

    eldict ={}
    
    for ielem in range(start,end):
        eltype = fypymesh.conn[ielem-1][-1]
        gdofn  = fypymesh.gdofn
        # elem   = getelem(eltype,ninteg,gdofn)
        
        if ( (hh := (eltype+str(ninteg))) not in eldict):
            elem = getelem(eltype,ninteg,gdofn)
            eldict[hh] = elem
        else:
            elem   = eldict[hh]

        # elem = copy.deepcopy(elem)
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

        with treduc:
            karr.extend(list(elem.kdata))
            krow.extend(list(elem.krow))
            kcol.extend(list(elem.kcol))
            rhsarr.extend(list(elem.fdata))
            rhsrow.extend(list(elem.frow))
            rhscol.extend(list(elem.fcol))
        reduction_time += treduc.elapsed

    return (karr,krow,kcol,rhsarr,rhsrow,rhscol)

    '''
    with treduc:
        kk   = sparse.coo_matrix((karr,(krow,kcol)),shape=(gdofn,gdofn),dtype='float64');
        rhs  = sparse.coo_matrix((rhsarr,(rhsrow,rhscol)),shape=(gdofn,1),dtype='float64');

    #print('Final reduction time = ',treduc.elapsed)
    reduction_time += treduc.elapsed
    
    return (kk,rhs,reduction_time)
    '''

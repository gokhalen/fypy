from scipy import sparse
import scipy.sparse.linalg
import numpy as np
import time

from ..libmesh import *

# get elements
from .assutils import *

from typing import Union,Tuple


ss         = sparse.coo_matrix
TOUTASS    = Tuple[ss,ss]

def assembly(fypymesh:FyPyMesh)->TOUTASS:
    gdofn  = fypymesh.gdofn
    ninteg = fypymesh.ninteg
    
    # initialize zero global matrix and global rhs vector 
    data = (0.0,); row = (0,); col = (0,); tt = (data,(row,col))
    kk  = sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64');
    rhs = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64');

    karr   = np.asarray([0]); krow   = np.asarray([0]); kcol   = np.asarray([0])
    rhsarr = np.asarray([0]); rhsrow = np.asarray([0]); rhscol = np.asarray([0])

    eldict = {}

    scipy_time = 0.0
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

        t1   = time.perf_counter()
        #kk  += elem.kmatrix
        #rhs += elem.rhs
        
        
        karr   = np.concatenate([karr,elem.kdata])
        krow   = np.concatenate([krow,elem.krow])
        kcol   = np.concatenate([kcol,elem.kcol])
        rhsarr = np.concatenate([rhsarr,elem.fdata])
        rhsrow = np.concatenate([rhsrow,elem.frow])
        rhscol = np.concatenate([rhscol,elem.fcol])
        

        t2   = time.perf_counter()
        scipy_time += (t2-t1)
        
    
    t1   = time.perf_counter()
    kk   = sparse.coo_matrix((karr,(krow,kcol)),shape=(gdofn,gdofn),dtype='float64');
    rhs  = sparse.coo_matrix((rhsarr,(rhsrow,rhscol)),shape=(gdofn,1),dtype='float64');
    t2   = time.perf_counter()
    scipy_time += (t2-t1)

    return (kk,rhs,scipy_time)

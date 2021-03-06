import scipy.sparse.linalg
import numpy as np
import sys

from timerit import Timer
from scipy import sparse

# from 
from ..libmesh import *
# get elements
from .assutils import *

from typing import Union,Tuple


ss         = sparse.coo_matrix
TOUTASS    = Tuple[ss,ss]

def assembly_list(fypymesh:FyPyMesh,dummy1,dummy2)->TOUTASS:
    # dummy arguments to ensure conistent interface across assembly routines
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
    
    for ielem in range(1,fypymesh.nelem+1):
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

        #kk  += elem.kmatrix
        #rhs += elem.rhs
        with treduc:
            karr.extend(list(elem.kdata))
            krow.extend(list(elem.krow))
            kcol.extend(list(elem.kcol))
            rhsarr.extend(list(elem.fdata))
            rhsrow.extend(list(elem.frow))
            rhscol.extend(list(elem.fcol))
        reduction_time += treduc.elapsed
        
    with treduc:
        kk   = sparse.coo_matrix((karr,(krow,kcol)),shape=(gdofn,gdofn),dtype='float64');
        rhs  = sparse.coo_matrix((rhsarr,(rhsrow,rhscol)),shape=(gdofn,1),dtype='float64');

    #print('Final reduction time = ',treduc.elapsed)
    reduction_time += treduc.elapsed
    
    return (kk,rhs,reduction_time)




    

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

def assembly_strain(fypymesh:FyPyMesh,dummy1,dummy2)->TOUTASS:
    # dummy arguments to ensure conistent interface across assembly routines
    gdofn      = fypymesh.gdofn
    ninteg     = fypymesh.ninteg
    straindofn = fypymesh.nnodes*fypymesh.ndofn
    
    # initialize zero global matrix and global rhs vector 
    data = (0.0,); row = (0,); col = (0,); tt = (data,(row,col))
    #kk  = sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64');
    #rhs = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64');

    marr   = [0]; mrow   = [0]; mcol   = [0]
    exxarr = [0]; exxrow = [0]; exxcol = [0]
    eyyarr = [0]; eyyrow = [0]; eyycol = [0]
    exyarr = [0]; exyrow = [0]; exycol = [0]    

    # we refer to the process of combining element matrices and vectors as 'reduction'
    reduction_time = 0.0
    treduc         = Timer('FyPy Assembly Strain (Reduction) Timer',verbose=0)

    eldict ={}
    
    for ielem in range(1,fypymesh.nelem+1):
        eltype = fypymesh.conn[ielem-1][-1]
        gdofn  = fypymesh.gdofn
        
        if ( (hh := (eltype+str(ninteg))) not in eldict):
            elem = getelem(eltype,ninteg,gdofn)
            eldict[hh] = elem
        else:
            elem   = eldict[hh]

        # nn is list of nodes 
        *nn,   =  fypymesh.conn[ielem-1][0:-1]

        # set coord
        coord     = np.asarray([ fypymesh.coord[n-1]  for n in nn])
        solution  = np.asarray([ fypymesh.solution[n-1] for n in nn])
        ideqnmass = np.asarray([ fypymesh.ideqnmass[n-1] for n in nn])

#       prop   = np.asarray([ fypymesh.prop[n-1]   for n in nn])
#       bf     = np.asarray([ fypymesh.bf[n-1]     for n in nn])
#       pforce = np.asarray([ fypymesh.pforce[n-1] for n in nn])
#       dirich = np.asarray([ fypymesh.dirich[n-1] for n in nn])
#       trac   = np.asarray([ fypymesh.trac[n-1]   for n in nn])
#       ideqn  = np.asarray([ fypymesh.ideqn[n-1]  for n in nn])

        # call the setdata method to initialize the element
        # note we're initializing only the data that is required to compute mass matrix
        elem.setdata(coord=coord,solution=solution,ideqnmass=ideqnmass)

        # compute mass
        elem.compute_mass()

        #kk  += elem.kmatrix
        #rhs += elem.rhs
        with treduc:
            marr.extend(list(elem.mdata))
            mrow.extend(list(elem.mrow))
            mcol.extend(list(elem.mcol))
            #rhsarr.extend(list(elem.fdata))
            #rhsrow.extend(list(elem.frow))
            #rhscol.extend(list(elem.fcol))
        reduction_time += treduc.elapsed
        
    with treduc:
        mm   = sparse.coo_matrix((marr,(mrow,mcol)),shape=(straindofn,straindofn),dtype='float64');
        fexx = sparse.coo_matrix((exxarr,(exxrow,exxcol)),shape=(straindofn,1),dtype='float64');
        feyy = sparse.coo_matrix((eyyarr,(eyyrow,eyycol)),shape=(straindofn,1),dtype='float64');
        fexy = sparse.coo_matrix((exyarr,(exyrow,exycol)),shape=(straindofn,1),dtype='float64');

    #print('Final reduction time = ',treduc.elapsed)
    reduction_time += treduc.elapsed

    '''
    md = mm.todense()
    for irow in range(9):
        for icol in range(9):
            print(f' {md[irow,icol]:0.2f}',sep='  ',end='')

        print('\n')
    '''
    
    return (mm,fexx,feyy,fexy,reduction_time)




    

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
        nnodes = fypymesh.nnodes
        
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

        # call the setdata method to initialize the element
        # note we're initializing only the data that is required to compute mass matrix
        elem.setdata(coord=coord,solution=solution,ideqnmass=ideqnmass)

        # compute mass
        elem.compute_mass_and_strain_forc()


        with treduc:
            marr.extend(list(elem.mdata))
            mrow.extend(list(elem.mrow))
            mcol.extend(list(elem.mcol))
            # 
            exxarr.extend(list(elem.exxrhs))
            exxrow.extend(list(elem.strainrow))
            exxcol.extend(list(elem.straincol))
            # 
            eyyarr.extend(list(elem.eyyrhs))
            eyyrow.extend(list(elem.strainrow))
            eyycol.extend(list(elem.straincol))
            #
            exyarr.extend(list(elem.exyrhs))
            exyrow.extend(list(elem.strainrow))
            exycol.extend(list(elem.straincol))
            
        reduction_time += treduc.elapsed
        
    with treduc:
        mm   = sparse.coo_matrix((marr,(mrow,mcol)),shape=(nnodes,nnodes),dtype='float64');
        fexx = sparse.coo_matrix((exxarr,(exxrow,exxcol)),shape=(nnodes,1),dtype='float64');
        feyy = sparse.coo_matrix((eyyarr,(eyyrow,eyycol)),shape=(nnodes,1),dtype='float64');
        fexy = sparse.coo_matrix((exyarr,(exyrow,exycol)),shape=(nnodes,1),dtype='float64');


    #print('Final reduction time = ',treduc.elapsed)
    reduction_time += treduc.elapsed
    
    '''
    print('-'*80)
    md = mm.todense()
    print('mm at the end of assembly_strain')
    print('[',end='')
    for irow in range(9):
        print('[',end='')
        for icol in range(9):
            print(f' {md[irow,icol]:0.2f},',sep=',',end='')

        print('],')
    print(']')
    print('-'*80)
    fexxd = fexx.todense()
    print('exx rhs at the end of assembly_strain')
    for irow in range(9):
        print(fexxd[irow,0],', ',end='')

    print('-'*80)
    '''
    
    return (mm,fexx,feyy,fexy,reduction_time)




    

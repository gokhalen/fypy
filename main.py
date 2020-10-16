# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:31:21 2020

@author: Nachiket Gokhale
"""

import sys,time
import cProfile
import argparse 

from src.libmesh   import *
from src.libassem  import *
from src.libsolve  import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FYPY: A finite element code written in Python')
    
    parser.add_argument('--nprocs',help='number of processes to use',required=False,type=int,default=1)
    parser.add_argument('--chunksize',help='chunksize to use',required=False,type=int,default=1)
    parser.add_argument('--inputfile',help='input json file',required=False,type=str,default='data.json.in')
    parser.add_argument('--outputfile',help='output json file',required=False,type=str,default='data.json.out')
    parser.add_argument('--partype',help='parallelization type: poolmap or async',required=False,
                        type=str,default='poolmap',choices=['poolmap','async'])
    solverlist = ['spsolve','bicg','bicgstab','cg','cgs','gmres','lgmres','minres','qmr','gcrotmk']
    solverstr  = str(solverlist)
    parser.add_argument('--solvertype',help=f'choose from: {solverstr}',required=False,type=str,default='spsolve',choices=solverlist)
    parser.add_argument('--profile',help=f'runs the assembly through the profiler cProfile',
                        required=False,type=str,default='False',choices=['True','False'])



    args = parser.parse_args()
    
    meshfile    = args.inputfile
    outfile     = args.outputfile
    nprocs      = args.nprocs
    chunksize   = args.chunksize
    partype     = args.partype
    solvertype  = args.solvertype
    profileflag = args.profile 
    
     
    start_time = time.perf_counter()
    
    print('FYPY: FYnite elements in PYthon ...executing fypy/main.py ')

    start_pre  = time.perf_counter()
    
    fypymesh = FyPyMesh();
    fypymesh.json_read(meshfile)
    
    end_pre = time.perf_counter()

    
    start_assem = time.perf_counter()
    
    # create stiffness matrix and rhs
    if ( nprocs == 1 ):
        if ( profileflag == 'True'):
            cProfile.run('kk,rhs,scipy_time = assembly(fypymesh)')
        if (profileflag == 'False'):
            kk,rhs,scipy_time = assembly(fypymesh)
    else:
        if (partype == 'poolmap'):
            print('Parallel (Mapped) assembly started..')
            kk,rhs,scipy_time = assembly_par(fypymesh,nprocs,chunksize)
        if (partype == 'async'):
            print('Parallel (Async) assembly started..')
            kk,rhs,scipy_time = assembly_async(fypymesh,nprocs,chunksize)
    
    end_assem = time.perf_counter()
        
    # then solver
    start_sol = time.perf_counter()
    solver    = FyPySolver(kk,rhs);
    solution  = solver.solve(solvertype)
    end_sol   = time.perf_counter()

    start_out = time.perf_counter()
    # then create output data    
    fypymesh.make_solution_from_rhs(solution)
    fypymesh.make_output(outfile)
    end_out = time.perf_counter()


    end_time = time.perf_counter()
    
    total_time = end_time  - start_time
    pre_time   = end_pre   - start_pre  
    assem_time = end_assem - start_assem
    solve_time = end_sol   - start_sol  
    out_time   = end_out   - start_out  

    print(f'{total_time=},\n{pre_time=},\n{assem_time=},\n{solve_time=},\n{out_time=},\n{scipy_time=}\n')
    print(f'{pre_time/total_time =},\n{assem_time/total_time =},\n'
          f'{scipy_time/total_time=},\n'
          f'{solve_time/total_time =},\n{out_time/total_time =},\n')
    


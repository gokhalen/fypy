# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:31:21 2020

@author: Nachiket Gokhale
"""

import sys,time
import cProfile
import argparse

from timerit import Timer

# fypy imports 
from src.libmesh   import *
from src.libassem  import *
from src.libsolve  import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FyPy: A finite element code written in Python')
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
    
     
    ttotal = Timer('FyPy Total timer',verbose=0)
    with ttotal:
        print('-'*80)
        print(f'FyPy: Fynite elements in Python3 written by Nachiket Gokhale gokhalen@gmail.com')
        print(f'-'*80)
        print(f'Solving using {solvertype}')
                

        tpre = Timer('Preprocessing timer',verbose=0)
        with tpre:
            fypymesh = FyPyMesh();
            fypymesh.json_read(meshfile)

        tassem = Timer(label='FyPy Assembly timer',verbose=0)
        # create stiffness matrix and rhs
        with tassem:
            if ( nprocs == 1 ):
                if ( profileflag == 'True'):
                    cProfile.run('kk,rhs,reduction_time = assembly(fypymesh)')
                if (profileflag == 'False'):
                    kk,rhs,reduction_time = assembly(fypymesh)
            else:
                if (partype == 'poolmap'):
                    print('Parallel (Mapped) assembly started..')
                    kk,rhs,reduction_time = assembly_par(fypymesh,nprocs,chunksize)
                if (partype == 'async'):
                    print('Parallel (Async) assembly started..')
                    kk,rhs,reduction_time = assembly_async(fypymesh,nprocs,chunksize)

        tsolve = Timer('FyPy: Solver timer',verbose=0)
        with tsolve:
            solver    = FyPySolver(kk,rhs);
            solution  = solver.solve(solvertype)

        tout = Timer('Output timer',verbose=0)
        with tout:
            fypymesh.make_solution_from_rhs(solution)
            fypymesh.make_output(outfile)

    digits = 3
    
    print('-'*80)
    print(f'Preprocessing time \t= {tpre.elapsed:0.{digits}f}s \t {(tpre.elapsed/ttotal.elapsed)*100:0.{digits}f} %')    
    print(f'Assembly time \t\t= {tassem.elapsed:0.{digits}f}s  \t {(tassem.elapsed/ttotal.elapsed)*100:0.{digits}f}%')
    print(f'Solver time \t\t= {tsolve.elapsed:0.{digits}f}s \t {(tsolve.elapsed/ttotal.elapsed)*100:0.{digits}f} %')
    print(f'Output time \t\t= {tout.elapsed:0.{digits}f}s \t {(tout.elapsed/ttotal.elapsed)*100:0.{digits}f} %')
    print(f'Total time \t\t= {ttotal.elapsed:0.{digits}f}s\t {(ttotal.elapsed/ttotal.elapsed)*100:0.{digits}f} %')
    print('-'*80)
    print(f'Reduction time (approx for multiproc) = {reduction_time:0.{digits}f}s, \
          {(reduction_time/ttotal.elapsed)*100:0.{digits}f}% (of Total time) ')
    print('-'*80)
    print('Goodbye!')
    print('-'*80)


    


    


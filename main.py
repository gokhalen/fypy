# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:31:21 2020

@author: Nachiket Gokhale
"""

import sys
import cProfile
import argparse

from timerit import Timer

# fypy imports 
from src.libmesh   import *
from src.libassem  import *
from src.libsolve  import *
from src.libio     import *


if __name__ == '__main__':

    args=getargs()
    
    ttotal = Timer('FyPy Total timer',verbose=0)
    with ttotal:
        welcome()
        print(f'Solving using {args.solvertype}')
                

        tpre = Timer('Preprocessing timer',verbose=0)
        with tpre:
            fypymesh = FyPyMesh();
            fypymesh.json_read(args.inputfile)

        # create stiffness matrix and rhs
        tassem = Timer(label='FyPy Assembly timer',verbose=0)
        with tassem:
            fassem = eval(f'assembly_{args.partype}')
            print(f'{args.partype} assembly started...'.capitalize())
            if ( args.profile == 'True'):
                cProfile.run('kk,rhs,reduction_time = fassem(fypymesh,args.nprocs,args.chunksize)')
            else:
                kk,rhs,reduction_time = fassem(fypymesh,args.nprocs,args.chunksize)
                
        tsolve = Timer('FyPy: Solver timer',verbose=0)
        with tsolve:
            solver    = FyPySolver(kk,rhs);
            solution  = solver.solve(args.solvertype)

        tout = Timer('Output timer',verbose=0)
        with tout:
            fypymesh.make_solution_from_rhs(solution)
            fypymesh.make_output(args.outputfile)

    printtime(tpre=tpre.elapsed,tassem=tassem.elapsed,tsolve=tsolve.elapsed,
              tout=tout.elapsed,ttotal=ttotal.elapsed,treduc=reduction_time,
              digits=3)

    
    goodbye()


    


    


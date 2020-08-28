# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:31:21 2020

@author: aa
"""

import sys


from src.libmesh.fypymesh    import *
from src.libassem.assembly   import *
from src.libsolve.fypysolver import *

if __name__ == '__main__':
    print('FYnite elements in PYthon ...executing fypy/main.py ')
    
    # check if atleast one argument (apart from the name) is supplied
    if ( len(sys.argv) < 2):
        print('Usage: main.py <meshfile>')
        sys.exit(1)
    
    meshfile=sys.argv[1]
        
    # create mesh object 
    fypymesh = FyPyMesh();
    fypymesh.json_read(meshfile)

    # create stiffness matrix and rhs
    kk,rhs = assembly(fypymesh)
        
    # then solver
    solver    = FyPySolver(kk,rhs);
    solution  = solver.solve('bicg') 

    fypymesh.make_solution_from_rhs(solution)

    # then create output data
    breakpoint()
    # create output files
    


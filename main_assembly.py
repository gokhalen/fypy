# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:31:21 2020

@author: aa
"""

import sys,time
# import cProfile


from src.libmesh.fypymesh    import *
from src.libassem.assembly   import *
from src.libassem.assembly2   import *
from src.libsolve.fypysolver import *


if __name__ == '__main__':
    
    start_time = time.perf_counter()
    
    print('FYnite elements in PYthon ...executing fypy/main.py ')

    start_pre  = time.perf_counter()
    
    # check if atleast one argument (apart from the name) is supplied
    if ( len(sys.argv) < 2):
        print('Usage: main.py <meshfile>')
        sys.exit(1)

        
    meshfile = sys.argv[1]
    outfile  = meshfile.strip('in') + 'out'
        
    # create mesh object 
    fypymesh = FyPyMesh();
    fypymesh.json_read(meshfile)
    
    end_pre = time.perf_counter()

    
    start_assem = time.perf_counter()
    
    # create stiffness matrix and rhs
    # cProfile.run('kk,rhs = assembly(fypymesh)')
    kk,rhs = assembly2(fypymesh)
    end_assem = time.perf_counter()
        
    # then solver
    start_sol = time.perf_counter()
    solver    = FyPySolver(kk,rhs);
    solution  = solver.solve('bicg')
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

    print(f'{total_time=},{pre_time=},{assem_time=}{solve_time=}{out_time=}')
    print(f'{pre_time/total_time =} {assem_time/total_time =} { solve_time/total_time =} {out_time/total_time =}')
    

import sys,time
from src.fypy import FyPy

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

    fypy = FyPy(meshfile)

    end_pre = time.perf_counter()

    start_assem = time.perf_counter()
    fypy.assembly()
    end_assem = time.perf_counter()
    
    start_sol = time.perf_counter()
    fypy.solve()
    end_sol   = time.perf_counter()

    
    start_out = time.perf_counter()
    fypy.output()
    end_out = time.perf_counter()
    end_time = time.perf_counter()

    total_time = end_time  - start_time
    pre_time   = end_pre   - start_pre  
    assem_time = end_assem - start_assem
    solve_time = end_sol   - start_sol  
    out_time   = end_out   - start_out  

    print(f'{total_time=},{pre_time=},{assem_time=}{solve_time=}{out_time=}')
    print(f'{pre_time/total_time =} {assem_time/total_time =} { solve_time/total_time =} {out_time/total_time =}')


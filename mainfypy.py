import sys,time

from src.fypy  import FyPy
from timerit   import Timer
from src.libio import *

if __name__ == '__main__':

    args = getargs()
    ttotal = Timer('FyPy Total timer',verbose=0)
    with ttotal:
        welcome()
        print(f'Solving using {args.solvertype}')
        
        tpre = Timer('Preprocessing timer',verbose=0)
        with tpre:
            fypy = FyPy(args)
            fypy.preprocess('0')
        
        tassem = Timer(label='FyPy Assembly timer',verbose=0)
        with tassem:
            print(f'{args.partype} assembly started...'.capitalize())
            fypy.assembly()
            reduction_time = fypy.reduction_time

        tsolve = Timer('FyPy: Solver timer',verbose=0)
        with tsolve:
            fypy.solve()

        tout = Timer('Output timer',verbose=0)
        with tout:
            fypy.output()
            fypy.postprocess('0')

    printtime(tpre=tpre.elapsed,tassem=tassem.elapsed,tsolve=tsolve.elapsed,
              tout=tout.elapsed,ttotal=ttotal.elapsed,treduc=reduction_time,
              digits=3)
    goodbye()

        

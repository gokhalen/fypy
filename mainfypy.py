import sys,time

from src.fypy  import FyPy
from timerit   import Timer
from src.libio import *

if __name__ == '__main__':

    args = getargs()
    ttotal = Timer('FyPy Total timer',verbose=0)
    with ttotal:
        welcome()
        fypy = FyPy(meshfile)
        fypy.assembly()
        fypy.solve()
        fypy.output()


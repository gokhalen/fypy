from .src.libmesh  import *
from .src.libassem import *
from .src.libsolve import *

class FyPy():
    def __init__(self,fin):
        self.fypymesh = FyPyMesh()
        self.fypymesh.json_read(fin)
        self.outfile  = fin.strip('in') + 'out'

    def assembly(self):
        self.kk,self.rhs = assembly(self.fypymesh)

    def solve(self,method='bicg'):
        solver = FyPySolver(self.kk,self.rhs);
        self.solution = solver.solve(method)

    def output(self):
        self.fypymesh.make_solution_from_rhs(self.solution)
        self.fypymesh.make_output(self.outfile)

    

    

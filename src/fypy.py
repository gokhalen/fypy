import cProfile
from .libmesh  import *
from .libassem import *
from .libsolve import *

class FyPy():
    def __init__(self,args):
        self.args = args
        self.fypymesh = FyPyMesh(args.inputdir,args.outputdir)
        self.fypymesh.json_read(self.args.inputfile)


    def assembly(self):
        fassem = eval(f'assembly_{self.args.partype}')
        if ( self.args.profile == 'True'):
            cProfile.run('kk,rhs,reduction_time = fassem(self.fypymesh,self.args.nprocs,self.args.chunksize)')
        else:
            self.kk,self.rhs,self.reduction_time = fassem(self.fypymesh,self.args.nprocs,self.args.chunksize)

    def solve(self,method='bicg'):
        solver = FyPySolver(self.kk,self.rhs);
        self.solution = solver.solve(self.args.solvertype)

    def output(self):
        self.fypymesh.make_solution_from_rhs(self.solution)
        self.fypymesh.make_output(self.args.outputfile)

    def postprocess(self,suffix):
        self.fypymesh.postprocess(suffix)


    def doeverything(self,suffix):
        self.assembly()
        self.solve()
        self.output()
        self.postprocess(suffix)
        
        
    

    

    

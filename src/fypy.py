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
        # profiling doesn't seem to work in class
        self.fassem = eval(f'assembly_{self.args.partype}')
        if ( self.args.profile == 'True'):
            cProfile.runctx('self.kk,self.rhs,self.reduction_time = self.fassem(self.fypymesh,self.args.nprocs,self.args.chunksize)',globals(),locals())
        else:
            self.kk,self.rhs,self.reduction_time = self.fassem(self.fypymesh,self.args.nprocs,self.args.chunksize)

    def solve(self,method='bicg'):
        solver = FyPySolver(self.kk,self.rhs);
        self.solution = solver.solve(self.args.solvertype)

    def output(self):
        self.fypymesh.make_solution_from_rhs(self.solution)
        self.fypymesh.make_output(self.args.outputfile)

    def preprocess(self,suffix):
        self.fypymesh.preprocess(suffix)

    def postprocess(self,suffix):
        self.fypymesh.postprocess(suffix)
        self.fypymesh.postprocess_pv(suffix)


    def doeverything(self,suffix):
        self.assembly()
        self.solve()
        self.output()
        self.postprocess(suffix)
        
        
    

    

    

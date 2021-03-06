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
        self.fassem = eval(f'assembly_{self.args.partype}')
        if ( self.args.profile == 'True'):
            cProfile.runctx('self.kk,self.rhs,self.reduction_time = self.fassem(self.fypymesh,self.args.nprocs,self.args.chunksize)',globals(),locals())
        else:
            self.kk,self.rhs,self.reduction_time = self.fassem(self.fypymesh,self.args.nprocs,self.args.chunksize)


    def assembly_str(self):
        if ( self.args.profile == 'True'):
            cProfile.runctx('self.mm,self.fexx,self.feyy,self.fexy,self.reduction_strain = assembly_strain(self.fypymesh,self.args.nprocs,self.args.chunksize)',globals(),locals())
        else:
            self.mm,self.fexx,self.feyy,self.fexy,self.reduction_strain = assembly_strain(self.fypymesh,self.args.nprocs,self.args.chunksize)


    def solve(self,method='bicg'):
        solver = FyPySolver(self.kk,self.rhs);
        self.solution = solver.solve(self.args.solvertype)
        self.fypymesh.make_solution_from_rhs(self.solution)


    def solve_strain(self):
        # exx
        solver   = FyPySolver(self.mm,self.fexx)
        self.exx = solver.solve(self.args.solvertype)
        self.fypymesh.exx = self.exx
        
        # eyy
        solver   = FyPySolver(self.mm,self.feyy)
        self.eyy = solver.solve(self.args.solvertype)
        self.fypymesh.eyy = self.eyy

        # eyy
        solver   = FyPySolver(self.mm,self.fexy)
        self.exy = solver.solve(self.args.solvertype)
        self.fypymesh.exy = self.exy
        
        #print('fypy.py: exx in solve_strain...')
        #print(self.exx)
        #print(self.eyy)
        #print(self.exy)
        
    def output(self):
        self.fypymesh.make_output(self.args.outputfile)

    def preprocess(self,suffix):
        self.fypymesh.preprocess(suffix)

    def postprocess(self,suffix):
        self.fypymesh.postprocess(suffix)
        # Causes issue when running inside a script
        # 'maximum number of clients reachedFatal Python error: Segmentation fault'
        # solution is to downgrade vtk
        # self.fypymesh.postprocess_pv(suffix)

    def doeverything(self,suffix):
        self.assembly()
        self.solve()
        self.assembly_str()
        self.solve_strain()
        self.output()
        self.postprocess(suffix)

        
        
    

    

    

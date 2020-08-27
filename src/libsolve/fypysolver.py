import scipy.sparse.linalg
from scipy import sparse

solveratol = 1e-12

class FyPySolver():
    def __init__(self,kk,rhs):
        self.kk  = kk
        self.rhs = rhs.todense()

    def solve(self,solver):
        x,exitCode = scipy.sparse.linalg.bicg(self.kk,self.rhs,atol=solveratol)
        
        if (exitCode != 0):
            raise RuntimeError(f'Solver exited with error={exitCode}')
            
        return x

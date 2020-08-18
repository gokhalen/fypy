import numpy as np
from .elembase import *
from ..libinteg.integrate import *

class LinElas1D(ElemBase):

    def __init__(self,ninteg,gdofn):
        
        self.eltype   =  "linelas1d"                # identifier                                                  
        self.elnodes  = 2                           # number of nodes in the element                              
        self.elndofn  = 1                           # dofn at each node                                               
        self.ndime    = 1                           # dimension of the element                                        
        self.dimspace = 1                           # dimension of the space in which the element lives
        self.nprop    = 1                           # only 'stiffness' is considered a property. body force is not
        
        super().__init__(ninteg=ninteg,gdofn=gdofn)

    @staticmethod    
    def stiffness_kernel(gausspts,shape,data):
        # this has to return a matrix
        return np.asarray( ( (1,0), (0,1)  ))


    def rhs_kernel(self):
        # this has to return a vector
        pass
    
    def compute_stiffness(self):
        self.estiff = integrate_parent(self.stiffness_kernel,self.gg,self.ss,self.propinterp,self.jj)

    def compute_rhs_body_force():
        pass

    def compute_rhs_traction():
        pass

    def compute_rhs_dirichlet():
        pass

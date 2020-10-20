import numpy as np, copy
from .elembase import *

class LinElasTrac2D(ElemBase):

    def __init__(self,ninteg,gdofn):
        self.elnodes  = 2
        self.elndofn  = 2
        self.ndime    = 1
        self.dimspace = 2
        self.nprop    = 2  # lambda and mu

        super().__init__(ninteg=ninteg,gdofn=gdofn)

    def stiffness_kernel(self,gausspts,shape,jaco,prop):
        return np.zeros(self.edofn*self.edofn).reshape(self.edofn,self.edofn)
     
    def rhs_bf_kernel(self,gausspts,shape,jaco,bf):
        return np.zeros(self.edofn)

    def rhs_trac_kernel(self,gausspts,shape,jaco,trac):
        # the traction is \int N_A h_i  , which is mapped to a vector
        assert ( trac.shape == (2,)), 'shape of trac not correct in rhs_trac_kernel for linelastrac2d'
        
        N1 = shape.shape[0]
        N2 = shape.shape[1]

        h1 = trac[0]
        h2 = trac[1]
        
        return np.asarray([N1*h1,N1*h2,N2*h1,N2*h2])

    def rhs_point_force(self):
        return np.zeros(self.edofn)

    

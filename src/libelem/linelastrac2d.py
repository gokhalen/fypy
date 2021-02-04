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
        # this returns a zero matrix of shape (4,4)
        # this is made one dimensional by raveling in create_global_Kf and appended to the matrix list
        # this (4,4) matrix is not added to the (8,8) matrix coming from the stiffness_kernel
        # of LinElas2D
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

    def mass_kernel(self,gausspts,shape,jaco,prop):
        # see notes in stiffness_kernel
        return np.zeros(self.elnodes*self.elnodes).reshape(self.elnodes,self.elnodes)


    def make_strains(self,solution,ss):
        # from nodal displacements compute strains at every integration point
        nn  = self.ninteg
        exx = np.zeros((nn,),dtype='float64')
        eyy = np.zeros((nn,),dtype='float64')
        exy = np.zeros((nn,),dtype='float64')
        return (exx,eyy,exy)

    def strain_kernel(self,gausspts,shape,jaco,prop):
        # see notes is stiffness_kernel
        rhs = np.zeros((self.elnodes,),dtype='float64')
        return rhs

    

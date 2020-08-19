import numpy as np
from .elembase import *
from ..libinteg.integrate import *

class LinElas1D(ElemBase):

    def __init__(self,ninteg,gdofn):
        
        self.eltype   =  "linelas1d" # identifier                                                                 
        self.elnodes  = 2            # number of nodes in the element                                             
        self.elndofn  = 1            # dofn at each node                                                              
        self.ndime    = 1            # dimension of the element                                                       
        self.dimspace = 1            # dimension of the space in which the element lives                          
        self.nprop    = 1            # only 'stiffness' is considered a property. body force is not               
        
        super().__init__(ninteg=ninteg,gdofn=gdofn)

    def stiffness_kernel(self,gausspts,shape,jaco,prop):
        # this has to return a matrix
        # the integral to be evaluated is \int_{x1}^{x2} N_{A,x}k(x)N_{B,x} dx -> \int_{-1}^{1} n_{A,X} k(x) n_{B,X) jdet d\xi
        # the kernel is n_{A,X} k(x) n_{B,X) which is a matrix
        # since the multiplication by jdet and weight is taken care of outside

        msg = f'wrong data shape for material property in stiffness_kernel expected ({self.nprop},1) got {prop.shape}'
        assert (prop.shape == (self.nprop,)), msg

        kk = prop[0]

        N1x = jaco.gder[0][0]; N2x = jaco.gder[1][0]

        k11 = N1x*kk*N1x;
        k12 = N1x*kk*N2x;
        k21 = N2x*kk*N1x;
        k22 = N2x*kk*N2x;

        return np.asarray(( (k11,k12), (k21,k22)  ))

    
    def rhs_bf_kernel(self,gausspts,shape,jaco,bf):
        # this has to return a vector
        # the integral to be evalated is \int_{x1}^{x2} N_A b dx -> \int_{-1}^{1} n_A b(x) jdet d\xi
        # the kernel is n_A b(x) which is a vector
        # the multiplicaton by jdet and weight is taken care outside
        
        msg = f'wrong data shape for body force in stiffness_kernel expected (1,) got {bf.shape}'
        assert(bf.shape == (1,)), msg

        N1 = shape.shape[0] ; N2 = shape.shape[1]

        b1 = N1*bf[0]
        b2 = N2*bf[0]
        
        return np.asarray ( (b1,b2) )

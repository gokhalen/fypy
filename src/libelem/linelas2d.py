from .elembase import *

class LinElas2D(ElemBase):
    
    def __init__(self,ninteg,gdofn):
        self.eltype   = 'linelas2d'
        self.elnodes  = 4
        self.elndofn  = 2
        self.ndime    = 2
        self.dimspace = 2
        self.nprop    = 2

        super().__init__(ninteg=ninteg,gdofn=gdofn)
        
    def stiffness_kernel(self,gausspts,shape,jaco,prop):
        # returns a 8x8 matrix for now
        return np.arange(8*8).reshape(8,8)

    def rhs_bf_kernel(self,gausspts,shape,jaco,bf):
        # the body force is \int N_A b_i
        assert (bf.shape == (self.elndofn,)),'wrong shape for body force in rhs_bf_kernel in linelas2d'

        N1,N2,N3,N4 = shape.shape
        b1,b2 = bf

        v1 = N1*b1
        v2 = N1*b2
        v3 = N2*b1
        v4 = N2*b2
        v5 = N3*b1
        v6 = N3*b2
        v7 = N4*b1
        v8 = N4*b2
        
        return np.asarray([v1,v2,v3,v4,v5,v6,v7,v8])

    def rhs_trac_kernel(self,gausspts,shape,jaco,trac):
        # traction is computed by special trac elements
        # this method does not compute traction
        # therefore this method returns a zero vector 
        return np.zeros(self.edofn)

    def rhs_point_force(self):
        return copy.deepcopy(self.pforce.reshape(self.edofn))

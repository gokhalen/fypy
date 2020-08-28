from .elembase import *

class LinElas2D(ElemBase):
    
    def __init__(self,ninteg,gdofn):
        self.eltype   = 'linelastrac2d'
        self.elnodes  = 4
        self.elndofn  = 2
        self.ndime    = 2
        self.dimspace = 2
        self.nprop    = 2

        super().__init__(ninteg=ninteg,gdofn=gdofn)
        
    def stiffness_kernel(self,gausspts,shape,jaco,prop):
        # returns a 8x8 identity matrix for now 
        return np.eye(8)

    def rhs_bf_kernel(self,gausspts,shape,jaco,bf):
        # returns zero for now
        return np.zeros(self.edofn)

    def rhs_trac_kernel(self,gausspts,shape,jaco,trac):
        # traction is computed by special trac elements
        # this method does not compute traction
        # therefore this method returns a zero vector 
        return np.zeros(self.edofn)

    def rhs_point_force(self):
        return copy.deepcopy(self.pforce.reshape(self.edofn))

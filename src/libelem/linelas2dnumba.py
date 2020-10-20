from .elembase          import *
from .linelas2d         import *
from .linelas2dnumbalib import *

import numba as nb

class LinElas2DNumba(LinElas2D):
    
    def __init__(self,ninteg,gdofn):
        self.elnodes  = 4
        self.elndofn  = 2
        self.ndime    = 2
        self.dimspace = 2
        self.nprop    = 2

        super().__init__(ninteg=ninteg,gdofn=gdofn)
        
    # override elembases' compute_stiffness for fast numba implementation
    def compute_stiffness(self):
        kk  = np.zeros(8*8,dtype='float64').reshape(8,8)
        compute_stiffness_nb(self.ninteg,self.gg,self.ss,self.jj,self.prop,kk)
        self.estiff = kk


    
    

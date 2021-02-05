from .elembase             import *
from .linelas2d            import *
from .linelas2dnumbalib    import *
from .linelas2dnumbasrilib import *

import numba as nb
import numpy as np

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
        kk     = np.zeros(8*8,dtype='float64').reshape(8,8)
        kk_sri = np.zeros(8*8,dtype='float64').reshape(8,8)
        # need to break the namedtuples into arrays implicitly ordered by integration point
        tmp  = [s.shape for s in self.ss] ; shp  = np.asarray(tmp)
        tmp  = [ j.gder for j in self.jj] ; gder = np.asarray(tmp)
        tmp  = [ j.jdet for j in self.jj] ; jdet = np.asarray(tmp)
        compute_stiffness_nb(self.ninteg,self.gg.wts,shp,gder,jdet,self.prop,kk)
        self.estiff = kk



        
    

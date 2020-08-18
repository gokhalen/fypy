import numpy as np

from ..libelem.linelas1d import *
from ..libelem.linelas2d import *
from .test import *


class TestLibElem(TestFyPy):
    
    def test_linelas1d(self):
        # le1d => linear elasticity 1d
        le1d = LinElas1D(ninteg=3,gdofn=10)
        coord = np.zeros(6).reshape(2,3)
        
        coord[0][0] = 0.0; coord[0][1]=0.0; coord[0][2] = 0.0;
        coord[1][0] = 1.0; coord[1][1]=0.0; coord[1][2] = 0.0;

        prop  = np.arange(2).reshape(2,1)
        le1d.coord = coord
        le1d.getjaco()
        le1d.prop = prop
        le1d.interp_prop()
        le1d.compute_stiffness()
        print(le1d.estiff)


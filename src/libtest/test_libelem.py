import numpy as np

from ..libelem.linelas1d import *
from ..libelem.linelas2d import *
from .test import *


class TestLibElem(TestFyPy):
    
    def test_linelas1d(self):
        # le1d => linear elasticity 1d
        le1d = LinElas1D(ninteg=3,gdofn=10)
        coord = np.arange(6).reshape(2,3)
        prop  = np.arange(2).reshape(2,1)
        le1d.coord = coord
        le1d.getjaco()
        le1d.prop = prop
        le1d.interp_prop()
        


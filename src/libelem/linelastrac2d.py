import numpy as np, copy
from .elembase import *
from ..libinteg.integrate import *

class LinElasTrac2D(ElemBase):

    def __init__(self,ninteg,gdofn):
        self.eltype  = 'linelastrac2d'
        self.elnodes  = 2
        self.elndofn  = 2
        self.ndime    = 1
        self.dimspace = 2
        self.nprop    = 2  # lambda and mu

        super().__init__(ninteg=ninteg,gdofn=gdofn)

from .gausslegendre import *
from ..libshape.shape import *
from ..libshape.jacobian import *
import math
import numpy as np
from typing import Iterable,Any

def integrate_parent(finteg:Callable,gauss:Iterable,shape:Iterable,data:Iterable,jac:Iterable)->Any:
    # Integrates finteg over the domain [-1,1]^n
    # can use math.fsum to avoid truncation error but it doesn't play nice with numpy arrays

    # finteg: Callable to be integrated
    #         arguments: (gauss point, parent shape function and parent derivative at that point,other necessary data) 
    #         first 3 arguments of integrate are passed too finteg
    
    # gauss : Iterable yielding Integ namedtuple corresponding to each integration point
    # shape : Iterable yielding Shape namedtuple corresponding to each integration point
    # data  : Iterable yielding data for each integration point
    # jdet  : Iterable yielding Jacobian determinant for each integration point

    # wtjac is not passed to finteg because it does not need it to evaluate functions
    
    funcgauss = map(finteg,gauss.pts,shape,data)
    
    *wtjac,   = ( gg*jj.jdet  for gg,jj in zip(gauss.wts,jac) )
    
    return sum((fg*wt for fg,wt in zip(funcgauss,wtjac)))



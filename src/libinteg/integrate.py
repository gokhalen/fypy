from .gausslegendre import *
from ..libshape import *
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
    # data  : Iterable yielding data (required by finteg) for each integration point
    # jac   : Iterable yielding Jacobian namedtuple for each integration point

    # wtjac is not passed to finteg because it does not need it to evaluate functions

    # Feb 3 2021
    # gausspts are passed to the finteg callable, but it seems they're not used in most cases
    
    funcgauss = map(finteg,gauss.pts,shape,jac,data)
    
    *wtjac,   = ( gg*jj.jdet  for gg,jj in zip(gauss.wts,jac) )

    # the expression inside the sum is a generator
    return sum((fg*wt for fg,wt in zip(funcgauss,wtjac)))



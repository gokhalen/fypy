from .gausslegendre import *
from ..libshape.shape import *
from ..libshape.jacobian import *
import math

def integrate_parent(finteg:Callable,gausspts:Integ,shape:Shape,data,wtjac)->float:
    # Integrates finteg over the domain [-1,1]^n
    # math.fsum to avoid truncation error

    # finteg: Callable to be integrated
    #         arguments: (gauss point, parent shape function and parent derivative at that point,other necessary data) 
    #         first 3 arguments of integrate are passed too finteg 
    
    funcgauss = map(finteg,gausspts,shape,data)
    return math.fsum((fg*wt for fg,wt in zip(funcgauss,wtjac)))



from .gausslegendre import *
from ..libshape.shape import *
from ..libshape.jacobian import *
import math

def integrate_parent(finteg:Callable,gausspts:Integ,shape:Shape,data,wtjac)->float:
    # Integrates finteg over the domain [-1,1]^n
    # can use math.fsum to avoid truncation error but it doesn't play nice with numpy arrays

    # finteg: Callable to be integrated
    #         arguments: (gauss point, parent shape function and parent derivative at that point,other necessary data) 
    #         first 3 arguments of integrate are passed too finteg 
    
    funcgauss = map(finteg,gausspts,shape,data)
    return sum((fg*wt for fg,wt in zip(funcgauss,wtjac)))



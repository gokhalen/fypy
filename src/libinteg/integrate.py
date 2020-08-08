from .gausslegendre import *
from ..libshape.shape import *
import math

def integrate(finteg:Callable,gauss:Integ,shape:Shape,jacobian:Jacobian)->float:
    # Integrates finteg over the domain [-1,1]^n
    # math.fsum to avoid truncation error 
    return math.fsum(map(finteg,gauss,data))



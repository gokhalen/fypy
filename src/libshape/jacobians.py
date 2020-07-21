import numpy as np
from typing import List,Tuple

from ..libutils import Point



def jacobian1d(xx:Tuple(Point,Point))->Tuple[np.ndarray,float]:
    '''
    returns the Jacobian determinant and Jacobian of a 1D linear element living in 3D space
    '''
    p1=xx[0]
    p2=xx[1]
    # get the distance between the points 
    hh=p1.distance(p2)
    jacdet=0.5*hh
    return (np.array(jacdet),jacdet)

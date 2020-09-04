# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 13:56:06 2020

@author: Nachiket Gokhale

1D,2D,3D shape functions and derivatives over the parent domain
One can consider cacheing these shape function routines using @lru_cache
from functools because they are evaluated at the integration points
which do not change. But we can avoid this by calling these routines when the 
only element class is initialized

"""

import functools, numpy as np
from typing import List,Tuple
from collections import namedtuple

TF1  = Tuple[float]
TF2  = Tuple[float,float]
TF3  = Tuple[float,float,float]
# Generic tuple type containing arbitrary tuples
TGEN = Tuple[float,...]

# TDER = Tuple of Tuples e.g. ((N1_x,N1_y),(N2_x,N2_y)...)
Shape = namedtuple('Shape',['shape','der'])


def shape1d(xi:TF1) -> Shape:
    '''
    1D linear shape function and parent domain derivatives
    '''
    N1 = (1.0/2.0)*(1-xi[0])
    N2 = (1.0/2.0)*(1+xi[0])
    
    shape = np.asarray((N1,N2))
    der   = np.asarray(((-0.5,),(+0.5,)))
    return Shape(shape=shape,der=der)

def shape2d(xi:TF2)->Shape:
    '''
    these shape functions can be made faster by computing quantities
    like (1-x[1]) once and storing them
    '''
    N1 = (1.0/4.0)*(1-xi[0])*(1-xi[1])
    N2 = (1.0/4.0)*(1+xi[0])*(1-xi[1]) 
    N3 = (1.0/4.0)*(1+xi[0])*(1+xi[1])
    N4 = (1.0/4.0)*(1-xi[0])*(1+xi[1])
    
    shape = np.asarray((N1,N2,N3,N4))
    
    # dij = derivative of the ith shape function in the jth direction
    d11  = -(1.0/4.0)*(1-xi[1]) 
    d12  = -(1.0/4.0)*(1-xi[0])
    d21  =  (1.0/4.0)*(1-xi[1])
    d22  = -(1.0/4.0)*(1+xi[0])
    d31  =  (1.0/4.0)*(1+xi[1])
    d32  =  (1.0/4.0)*(1+xi[0])
    d41  = -(1.0/4.0)*(1+xi[1])
    d42  =  (1.0/4.0)*(1-xi[0])
    
    D1 = (d11,d12)
    D2 = (d21,d22)
    D3 = (d31,d32)
    D4 = (d41,d42)
    
    der = np.asarray((D1,D2,D3,D4))
       
    return Shape(shape=shape,der=der)
    
def shape3d(xi:TF3):
    pass


def getshape(ndim):
    assert ( ndim >= 1)

    if ( ndim == 1):
        return shape1d
    
    if ( ndim == 2):
        return shape2d



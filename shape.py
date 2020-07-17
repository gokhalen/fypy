# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 13:56:06 2020

@author: Nachiket Gokhale

1D,2D,3D shape functions and derivatives over the parent domain
Since shape functions are evaluated over the same integration points in 
parent domain, they are cached

"""

import functools

@functools.lru_cache
def shape1d(xi):
    '''
    1D linear shape function
    '''
    N1 = (1/2)*(1-xi)
    N2 = (1/2)*(1+xi)
    return (N1,N2)


@functools.lru_cache
def shape1d_der(xi):
    return (-0.5,0.5)

@functools.lru_cache
def shape2d(xi,eta):
    pass

@functools.lru_cache
def shape2d_der(xi,eta):
    pass
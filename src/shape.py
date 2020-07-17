# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 13:56:06 2020

@author: Nachiket Gokhale

1D,2D,3D shape functions and derivatives over the parent domain
Since shape functions are evaluated over the same integration points in 
parent domain, they are cached

"""

import functools
import typing

from typing import List,Tuple

TF1 = Tuple[float]
TF2 = Tuple[float,float]
TF3 = Tuple[float,float,float]

TF4 = Tuple[float,float,float,float]
TF8 = Tuple[float,float,float,float,float,float,float,float]

@functools.lru_cache
def shape1d(xi:TF1) -> TF2:
    '''
    1D linear shape function
    '''
    N1 = (1/2)*(1-xi)
    N2 = (1/2)*(1+xi)
    return (N1,N2)


@functools.lru_cache
def shape1d_der(xi:TF1)-> TF2:
    '''
    1D linear shape function derivatives
    '''
    return (-0.5,0.5)

@functools.lru_cache
def shape2d(xi:TF2)->TF4:
    print('hello')
    pass

@functools.lru_cache
def shape2d_der(xi:TF2):
    pass

@functools.lru_cache
def shape3d(xi:TF3):
    pass

@functools.lru_cache
def shape3d_der(xi:TF3):
    pass
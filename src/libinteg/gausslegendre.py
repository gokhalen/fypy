# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 12:46:34 2020

@author: Nachiket Gokhale

Creates Gauss-Legendre quadrature points in [-1,1] for 1D,2D and 3D elements
returns tuple(points,weights)

"""
from collections import namedtuple
from scipy.special import roots_legendre
from typing import Callable,Any
import itertools,functools, math, numpy as np

closetol=1e-12

# npoints is number of Gauss-Legendre integration points

Integ = namedtuple('Integ',['pts','wts'])

def gauss1d(npoints):
    # points and weights are np arrays, convert them into tuple
    p1d,w1d = roots_legendre(npoints)
    # need to get data structure right for consistency
    # pts must be an array of [x,y,z]
    # an elegant way is to use newaxis
    points=p1d[:,np.newaxis]
    weights = w1d
    return Integ(pts=points,wts=weights)


def gauss2d(npoints):
    p1d,w1d = gauss1d(npoints)
    # p1d is an array of arrays. need to be careful passing it to itertools.product
    # result will be array of tuples of arrays, which is not what we want
    tt=tuple([d for d in p1d.reshape(npoints)])
    it      = map(np.asarray,itertools.product(tt,tt))
    points  = np.asarray((*it,))
    
    it = map(math.prod,itertools.product(w1d,w1d))
    weights = np.asarray((*it,))
    return Integ(pts=points,wts=weights)     


def gauss3d(npoints):
    p1d,w1d = gauss1d(npoints)
    it      = map(np.asarray,itertools.product(p1d,repeat=3))
    points  = np.asarray((*it,))

    it = map(math.prod,itertools.product(w1d,repeat=3))
    weights = np.asarray((*it,))
    return Integ(pts=points,wts=weights) 

def gaussnd(ndim,npoints):
    p1d,w1d = gauss1d(npoints)
    # compute cartesian product of p1d. we first create arrays of integration points
    # and then an array of those arrays
    it      = map(np.asarray,itertools.product(p1d,repeat=ndim))
    points  = np.asarray((*it,))

    # compute integration weights by computing cartesian product of weights and
    # then taking the product of each tuple
    it      = itertools.product(w1d,repeat=ndim) 
    mm      = map(math.prod,it)
    weights = np.asarray((*mm,))
    
    return Integ(pts=points,wts=weights)


def getgauss(ndim,npoints):
    
    assert (ndim > 0)
    assert (npoints > 0)
    
    if   (ndim == 1):
        return gauss1d(npoints)
    elif (ndim == 2):
        return gauss2d(npoints)
    elif (ndim == 3):
        return gauss3d(npoints)
    else:
        return gaussnd(ndim,npoints)

    
#assert (gaussnd(ndim=3,npoints=3)==gauss3d(3)), 'gauss3d & gaussnd are inconsistent'
#assert (gaussnd(ndim=2,npoints=3)==gauss2d(3)), 'gauss2d & gaussnd are inconsistent'
#assert (np.allclose(gaussnd(ndim=1,npoints=3).wts,gauss1d(3).wts)), 'gauss1d & gaussnd are inconsistent'
#assert (np.allclose(gaussnd(ndim=1,npoints=3).pts,gauss1d(3).pts)), 'gauss1d & gaussnd are inconsistent'

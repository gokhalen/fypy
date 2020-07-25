# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 12:46:34 2020

@author: Nachiket Gokhale

Creates Gauss-Legendre quadrature points in [-1,1] for 1D,2D and 3D elements
returns tuple(points,weights)

"""

from scipy.special import roots_legendre
import itertools,functools

# npoints is number of Gauss-Legendre integration points

def gauss1d(npoints):
    # points and weights are np arrays, convert them into tuple
    points,weights = roots_legendre(npoints)
    return (tuple(points),tuple(weights))


def gauss2d(npoints):
    p1d,w1d = gauss1d(npoints)
    points  = tuple(itertools.product(p1d,p1d))
#   can also do 
#   weights = tuple(map(lambda x: x[0]*x[1], itertools.product(px,px)))
    weights = tuple(w1*w2 for w1,w2 in itertools.product(w1d,w1d))
    return (points,weights)     


def gauss3d(npoints):
    p1d,w1d = gauss1d(npoints)
    points  = tuple(itertools.product(p1d,repeat=3))
    weights = tuple(w1*w2*w3 for w1,w2,w3 in itertools.product(w1d,repeat=3))
    return (points,weights) 

def gaussnd(ndim,npoints):
    p1d,w1d = gauss1d(npoints)
    points  = tuple(itertools.product(p1d,repeat=ndim))
    it      = itertools.product(w1d,repeat=ndim) 
   
    def fprod(x):
#   fprod takes in an iterable and calculates it's product
        return functools.reduce(lambda a,b: a*b,x)
   
    weights = tuple(map(fprod,it))

#    can do it without fprod as well, just using lambdas, but it is confusing
#    it = itertools.product(w1d,repeat=ndim) 
#    weights2=tuple(map(lambda x: functools.reduce(lambda a,b: a*b,x),it))
    
    return (points,weights)


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

assert (gaussnd(ndim=3,npoints=3)==gauss3d(3)), 'gauss3d & gaussnd are inconsistent'
assert (gaussnd(ndim=2,npoints=3)==gauss2d(3)), 'gauss2d & gaussnd are inconsistent'
assert (gaussnd(ndim=1,npoints=3)==gauss1d(3)), 'gauss1d & gaussnd are inconsistent'

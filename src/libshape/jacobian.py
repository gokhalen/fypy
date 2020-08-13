import numpy as np
from typing import List,Tuple,Iterable
from collections import namedtuple

zerotol = 1e-12

# points are ALWAYS three dimensional
# jaco1d handles 1d elements living in 3d space, but jaco2d only handles 2d elements living in 2d space. z coordinate is ingnored for jaco2d.
# to handle 2d elements living in 3d space, I think one needs to change to coordinate system such that the new x and y axes are in the plane of the element
# and the new axis is perpendicular to it. In this new system one can evaluate the integrals.

# methods to evaluate jacobians, and jacobian determinants
# In each element, these routines have to be called for each integration point
# the element nodes are constnt for each element
# these routines can be used with map as follows
# *jj, = map(jaco1d,itertools.repeat(pp),shpder)
# where shpder is an iterable yielding derivatives of shapefunctions at point

# jaco is a matrix nsd*nsd and jdet is a scalar

Jaco=namedtuple('Jaco',['jaco','jdet','gder'])

def jaco1d(pp:np.ndarray,shpder:np.ndarray)->Jaco:
    # pp:     Iterable yielding two 3d points (np arrays)
    # shpder: derivatives of shape functions in parent domain

    # even though pp1 and pp2 live in 3d, we will treat pp1 as 0.0 and pp2 as norm(pp1-pp2)
    pp1 = pp[0]; pp2 = pp[1]

    # length of the element 
    ll   = np.linalg.norm(pp1-pp2)

    # number of jacobian determinants is the number of integration points
    jdet = ll/2.0

    assert ( jdet > zerotol ), f'jdet in is < {zerotol}'

    jaco = np.asarray( ((jdet,),) )

    # compute global derivatives
    gder1 = shpder[0][0]/jdet
    gder2 = shpder[1][0]/jdet

    gder  = np.asarray(( (gder1,),(gder2,) ))

    return Jaco(jdet=jdet,jaco=jaco,gder=gder)

def jaco2d(pp:Iterable,shpder)->Jaco:
    # pp:     Iterable yielding 3d points (np array)
    # shpder: Derivatives of shape functions in parent domain
    return Jaco(jdet=None,jaco=None)

def jaco3d():
    pass
    




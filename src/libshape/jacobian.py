import numpy as np
from typing import List,Tuple,Iterable
from collections import namedtuple

zerotol = 1e-12
# points are ALWAYS three dimensional
# methods to evaluate jacobians, and jacobian determinants
# In each element, these routines have to be called for each integration point
# the element nodes are constnt for each element
# these routines can be used with map as follows
# *jj, = map(jaco1d,itertools.repeat(pp),shpder)
# where shpder is an iterable yielding derivatives of shapefunctions at point

# jaco is a matrix nsd*nsd and jdet is a scalar
Jaco=namedtuple('Jaco',['jaco','jdet','gder'])

def jaco1d(pp:np.ndarray,shpder:np.ndarray)->Jaco:
    # pp:     Iterable yielding 3d points (np arrays)
    # shpder: derivatives of shape functions in parent domain
    # neglect negative jacobian det in this routine
    # (in linear transformations we can always map [a,b] to [-1,1] or [1,-1] without changing sign of jdet )

    # test integration for this, and check if it throws AssertionError

    pp1 = pp[0]; pp2 = pp[1]

    # length of the element 
    ll   = np.linalg.norm(pp1-pp2)
    jdet = ll/2.0
    assert ( jdet > zerotol ), f'jdet in is < {zerotol}'

    jaco = np.asarray( ((jdet,),) )

    # compute global derivatives
    gder1 = shpder[0][0]*jaco
    gder2 = shpder[1][0]*jaco

    gder  = np.asarray(( (gder1,),(gder2,) ))

    return Jaco(jdet=jdet,jaco=jaco,gder=gder)

def jaco2d(pp:Iterable,shpder)->Jaco:
    # pp:     Iterable yielding 3d points (np array)
    # shpder: Derivatives of shape functions in parent domain
    return Jaco(jdet=None,jaco=None)

def jaco3d():
    pass
    




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
    # shpder: derivatives of shape functions in parent domain at particular integration point

    # output:  jdet,jaco and gder (global derivatives) at a particular integration point

    # even though pp1 and pp2 live in 3d, we will treat pp1 as 0.0 and pp2 as norm(pp1-pp2)
    pp1 = pp[0]; pp2 = pp[1]

    # length of the element 
    ll   = np.linalg.norm(pp1-pp2)

    # number of jacobian determinants is the number of integration points
    jdet = ll/2.0

    assert ( jdet > zerotol ), f'jdet in jaco1d is < {zerotol} in jaco1d'

    jaco = np.asarray( ((jdet,),) )

    # compute global derivatives
    gder1 = shpder[0][0]/jdet
    gder2 = shpder[1][0]/jdet

    gder  = np.asarray(( (gder1,),(gder2,) ))

    return Jaco(jdet=jdet,jaco=jaco,gder=gder)

def jaco2d(pp:Iterable,shpder)->Jaco:
    # pp:     Iterable yielding 4 3d points (np array). the third coordinate (z) is ignored.
    # shpder: Derivatives of shape functions in parent domain at particular integration point
    # can also be done via xvec,yvec,zvec = zip(*pp)

    # output -> jdet,jaco and gder (global derivatives) at a particular integration point
    
    xvec = np.asarray([p[0] for p in pp])
    yvec = np.asarray([p[1] for p in pp])
    xshp = shpder[:,0]
    yshp = shpder[:,1]

    # derivative of x in direction 1
    x1 = np.dot(xvec,xshp)
    x2 = np.dot(xvec,yshp)
    y1 = np.dot(yvec,xshp)
    y2 = np.dot(yvec,yshp)

    jaco = np.asarray([[ x1, x2 ],[ y1, y2 ]])
    jdet = np.linalg.det(jaco)

    assert ( jdet > zerotol ), f'jdet in is < {zerotol} in jaco2d'

    jacoinv = (1.0/jdet)*np.asarray([ [ y2, -x2],  [ -y1, x1 ]  ])

    gder    = shpder@jacoinv
    
    return Jaco(jdet=jdet,jaco=jaco,gder=gder)

def jaco3d():
    pass
    



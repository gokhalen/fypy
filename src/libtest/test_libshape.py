import unittest,functools,math,itertools
import numpy as np

from typing import Callable,Iterable
from ..libshape.shape import *

closetol=1e-12
npallclose=functools.partial(np.allclose,atol=closetol)

class TestLibShape(unittest.TestCase):

    # tests for shape1d and shape2d .
    # input for both is a point represented by a tuple of floats
    # output is ( shape, der )
    # shape: (N1,N2...)
    # der  : ((N1x,N1y..),(N2x,N2y,...))

    def compare_shape(self,fshape:Callable,idata:Iterable,exout:Iterable):
        '''
        fshape: callable to be tested shape1d or shape2d
        idata:  points at which shape functions and their derivatives have to be tested
        exout:  expected output 
        actout: actual output
        '''
        *actout,=map(fshape,idata)
        for ipt,((ash,ad),(exsh,exd)) in enumerate(zip(actout,exout)):
            # ash, ad   = actual (a) test shape functions (sh) and derivatives (d)
            # exsh, exd = expected (ex) shape functions (sh) and derivatives (d)
            # convert to np
            ash,ad,exsh,exd  = map(np.asarray,(ash,ad,exsh,exd))
            # boolean arrays with True when actual == expected 
            shclose          = np.isclose(ash,exsh,atol=closetol)
            dclose           = np.isclose(ad,exd,atol=closetol)
            # shclose should not have any False elements
            
    
    def test_shape1d(self):
        # points for which the testing is to be performed
        pts=[]
        pts.append((-1,))
        pts.append((-0.5,))
        pts.append((0.0,))
        pts.append((0.5,))
        pts.append((1.0,))
        #pts=((-1,),(-0.5,),(0,),(+0.5,),(1,))

        # expected values of shape functions at that point
        nn=[]
        nn.append((1.0,0.0)) 
        nn.append((0.75,0.25))
        nn.append((0.5,0.5))
        nn.append((0.25,0.75))
        nn.append((0.0,1.0))

        # for 1d shape functions the derivatives are constant
        der=((-0.5,),(0.5,))
    
        # expected output
        *exout,=zip(nn,itertools.repeat(der))

        self.compare_shape(shape1d,pts,exout)

        


        



import unittest,functools,math,itertools
import numpy as np

from typing import Callable,Iterable,Union
from ..libshape.shape import *
from .test import *
from ..libutil.util import *

class TestLibShape(TestFyPy):

    # tests for shape1d and shape2d .
    # input for both is a point represented by a tuple of floats
    # output is ( shape, der )
    # shape: (N1,N2...)
    # der  : ((N1x,N1y..),(N2x,N2y,...))

    def compare_shape(self,fshape:Callable,pts:Iterable,exout:Iterable):
        '''
        fshape: callable to be tested: shape1d or shape2d
        pts:    points at which shape functions and their derivatives have to be tested
        exout:  expected output containing Shape named tuples 
        '''
        #  actout: actual output to be compard with 'exout' expected output
        *actout,             = map(fshape,pts)
        msgshp,msgder   = 'Mismatch in shape functions: ','Mismatch in shape function derivatives: '

        for i,(act,exp) in enumerate(zip(actout,exout)):
            boolshp,boolder = map(npclose,(act.shape,act.der),(exp.shape,exp.der))
            if ( not boolshp ):
                idx,aa,bb = get_mismatch(act.shape,exp.shape,closetol=closetol)
                msgshp    += make_mismatch_message(idx,aa,bb) + f'for {i}th entry in testing data'
                pass

            if ( not boolder ):
                print(f'{act=}',f'{exp=}')
                idx,aa,bb  = get_mismatch(act.der,exp.der,closetol=closetol)
                msgder    += make_mismatch_message(idx,aa,bb) + f'for {i}th entry in testing data'
                pass
            
            self.assertTrue(boolshp,msg=msgshp)
            self.assertTrue(boolder,msg=msgder)
    
    def test_shape1d(self):
        # points for which the testing is to be performed, note: points are tuples
        pts  = ((-1,),(-0.5,),(0.0,),(0.5,),(1.0,))
        # expected values for shape functions 
        shp  = ((1.0,0.),(0.75,0.25),(0.5,0.5),(0.25,0.75),(0.0,1.0))
        shp  = np.asarray(shp)

        # sanity test - should produce error in [2][1] shape function
        # shp[2][1] += 1
        
        # expected values of shape functions and derivatives
        # for 1d shape functions the derivatives are constant
        # since we're going to map Shape and shp, der
        # der needs to be an 3D array. i.e. on iterating over der we need to get 2D arrays
        # each of which is a derivative object
        
        der=((-0.5,),(0.5,))
        der=np.asarray(der)
        # this produces a 2D array need to multiply it by number of arrays
        der=(der,)*len(shp)
        der=np.asarray(der)

        # sanity test - should produce error in [3][1] component of der
        # der[3][1][0] +=1

        # or equivalently, with broadcasting
        # der = ((-0.5,),(0.5,))
        # der = np.asarray(der)
        # aa  = np.zeros(10).reshape(5,2,1)
        # aa += der

        #expected output
        *exout,=map(Shape,shp,der)

        self.compare_shape(shape1d,pts,exout)

        # sanity test - should produce error in [2][1] shape function
        # shp[2][1] += 1
        
        # sanity test - should produce error in [3][1] component of der
        der[3][1][0] +=1
        datamsg =  [" Shape Functions ", " Shapefunction Derivatives "]
        ptst    =  tuple([ (tt,) for tt in pts ])
        # ptst is a tuple of tuple of arguments to shape1d. Each tuple when unpacked yields the right argument  
        # self.compare_test_data(shape1d,ptst,None,None,exout,datamsg,data_supplied=True,optmsg="Testing 1D Shape Functions ")

        


        



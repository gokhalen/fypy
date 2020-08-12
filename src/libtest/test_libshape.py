import unittest,functools,math,itertools
import numpy as np

from typing import Callable,Iterable,Union
from ..libshape.shape import *
from ..libshape.jacobian import *
from .test import *
from ..libutil.util import *
from ..libinteg.gausslegendre import *


maxinteg = 10
class TestLibShape(TestFyPy):
    
    @staticmethod
    def global_1d_shape_der(x,x1,x2):
        # x1 and x2 are scalars
        N1x =  - 1/(x2 - x1)
        N2x =  + 1/(x2 - x1)
        return np.asarray(( (N1x,),(N2x) ))

    # tests for shape1d and shape2d .
    # input for both is a point represented by a tuple of floats
    # output is ( shape, der )
    # shape: (N1,N2...)
    # der  : ((N1x,N1y..),(N2x,N2y,...))

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

        #self.compare_shape(shape1d,pts,exout)

        # sanity test - should produce error in [2][1] shape function
        # shp[2][1] += 1
        
        # sanity test - should produce error in [3][1] component of der
        # der[3][1][0] +=1
        datamsg =  [" Shape Functions ", " Shapefunction Derivatives "]
        ptst    =  tuple([ (tt,) for tt in pts ])
        # ptst is a tuple of tuple of arguments to shape1d. Each tuple when unpacked yields the right argument
        self.compare_test_data(ftest=shape1d,fargs=ptst,truedata=exout,datamsg=datamsg,optmsg="Testing 1D Shape Functions ")

        
    
    def test_shape2d(self):
        # print('testing shape2d...')
        # pp - list of points (tuples) to be tested
        # ss - list of expected shape functions at each point (tuples)
        # dd - list of expected shape function derivatives at each point
        pp,ss,dd = [],[],[]
        pp.append((0.0,0.0))   ; ss.append(np.asarray((0.25,   0.25  , 0.25  ,  0.25   ))) # done
        pp.append((0.5,0.5))   ; ss.append(np.asarray((0.0625, 0.1875, 0.5625,  0.1875 ))) # done
        pp.append((-0.5,0.5))  ; ss.append(np.asarray((0.1875, 0.0625, 0.1875,  0.5625 )))
        pp.append((-0.5,-0.5)) ; ss.append(np.asarray((0.5625, 0.1875, 0.0625,  0.1875 )))
        pp.append((0.5,-0.5))  ; ss.append(np.asarray((0.1875, 0.5625, 0.1875,  0.0625 )))
        pp.append((-1,-1))     ; ss.append(np.asarray((1     ,    0.0,   0.0,    0.0   )))
        pp.append((1,-1))      ; ss.append(np.asarray((0     ,    1.0,   0.0,    0.0   )))
        pp.append((1,1))       ; ss.append(np.asarray((0     ,    0.0,   1.0,    0.0   )))
        pp.append((-1,1))      ; ss.append(np.asarray((0     ,    0.0,   0.0,    1.0   )))
        # should also work when points are appended as np arrays. # gaussian quadrature routines provide data in np arrays
        pp.append(np.asarray((0.3,0.7))) ;    ss.append(np.asarray([0.0525,0.0975,0.5525,0.2975]))
        pp.append(np.asarray((-0.25,0.85))) ; ss.append(np.asarray([0.046875,0.028125,0.346875,0.578125]))
        pp.append(np.asarray((-0.7,-0.15))) ; ss.append(np.asarray([0.48875,0.08625,0.06375,0.36125]))
        pp.append(np.asarray((0.19,-0.55))) ; ss.append(np.asarray([0.313875,0.461125,0.133875,0.091125]))


        # compute shape function derivatives (0.0,0.00)
        d1 = np.asarray((-0.25,-0.25)); d2 = np.asarray((0.25,-0.25)); d3 = np.asarray((0.25,0.25)); d4=np.asarray((-0.25,0.25))
        dd.append(np.asarray((d1,d2,d3,d4)))

        # (0.5,0.5)
        d1 = np.asarray((-0.125,-0.125)); d2 = np.asarray((0.125,-0.375)); d3 = np.asarray((0.375,0.375)); d4=np.asarray((-0.375,0.125))
        dd.append(np.asarray((d1,d2,d3,d4)))

        # (-0.5,0.5)
        d1 = np.asarray((-0.125,-0.375)); d2 = np.asarray((0.125,-0.125)); d3 = np.asarray((0.375,0.125)); d4=np.asarray((-0.375,0.375))
        dd.append(np.asarray((d1,d2,d3,d4)))

        # (-0.5,-0.5)
        d1 = np.asarray((-0.375,-0.375)); d2 = np.asarray((0.375,-0.125)); d3 = np.asarray((0.125,0.125)); d4=np.asarray((-0.125,0.375))
        dd.append(np.asarray((d1,d2,d3,d4)))

        # (0.5,-0.5)
        d1 = np.asarray((-0.375,-0.125)); d2 = np.asarray((0.375,-0.375)); d3 = np.asarray((0.125,0.375)); d4=np.asarray((-0.125,0.125))
        dd.append(np.asarray((d1,d2,d3,d4)))
        
        # (-1,-1)
        d1 = np.asarray((-0.5,-0.5)); d2 = np.asarray((0.5,0.0)); d3 = np.asarray((0.0,0.0)); d4=np.asarray((0.0,0.5))
        dd.append(np.asarray((d1,d2,d3,d4)))

        # (1,-1)
        d1 = np.asarray((-0.5,0.0)); d2 = np.asarray((0.5,-0.5)); d3 = np.asarray((0.0,0.5)); d4=np.asarray((0.0,0.0))
        dd.append(np.asarray((d1,d2,d3,d4)))

        # (1,1)
        d1 = np.asarray((0.0,0.0)); d2 = np.asarray((0.0,-0.5)); d3 = np.asarray((0.5,0.5)); d4=np.asarray((-0.5,0.0))
        dd.append(np.asarray((d1,d2,d3,d4)))

        # (-1,1)
        d1 = np.asarray((0.0,-0.5)); d2 = np.asarray((0.0,0.0)); d3 = np.asarray((0.5,0.0)); d4=np.asarray((-0.5,0.5))
        dd.append(np.asarray((d1,d2,d3,d4)))

        # (0.3,0.7)
        d1 = np.asarray((-0.075,-0.175)); d2 = np.asarray((0.075,-0.325)); d3 = np.asarray((0.425,0.325)); d4=np.asarray((-0.425,0.175))
        dd.append(np.asarray((d1,d2,d3,d4)))

        # (-0.25,0.85)
        d1 = np.asarray((-0.0375,-0.3125)); d2 = np.asarray((0.0375,-0.1875)); d3 = np.asarray((0.4625,0.1875)); d4=np.asarray((-0.4625,0.3125))
        dd.append(np.asarray((d1,d2,d3,d4)))
        
        # (-0.7,-0.15)
        d1 = np.asarray((-0.2875,-0.425)); d2 = np.asarray((0.2875,-0.075)); d3 = np.asarray((0.2125,0.075)); d4=np.asarray((-0.2125,0.425))
        dd.append(np.asarray((d1,d2,d3,d4)))

        # (0.19,-0.55)
        d1 = np.asarray((-0.3875,-0.2025)); d2 = np.asarray((0.3875,-0.2975)); d3 = np.asarray((0.1125,0.2975)); d4=np.asarray((-0.1125,0.2025))
        dd.append(np.asarray((d1,d2,d3,d4)))
        
        # expected output
        *exout, = map(Shape,ss,dd)

        # the iterable containing arguments, contains iterables which when expaned will be come arguments
        pts = tuple([(p,) for p in pp])

        # make sure shape functions sum to 1
        for i,s in enumerate(ss):
            tmp = np.sum(s)
            self.assertEqual(1,tmp,msg=f'{i}th shape functions do not sum to 1')

            
        datamsg=['Shape functions ','Derivatives of shape functions ']
        self.compare_test_data(ftest=shape2d,fargs=pts,truedata=exout,datamsg=datamsg,optmsg='Testing 2D shape functions...')

    def test_jaco1d(self):
        for ipoint in range(1,maxinteg):
            # generate random interval (defined by two points and a straight line joining them) to test.
            # The endpoints of the interval must not be same, hence the while and break
            while True:
                p1   = np.random.rand(3)
                p2   = np.random.rand(3)

                if ( np.linalg.norm(p1-p2) > zerotol ):
                    break;
                
            gg   = gauss1d(ipoint)
            mm   = map(shape1d,gg.pts)
            der  = [ ss.der for ss in mm]
            *jj, = map(jaco1d,itertools.repeat([p1,p2]),der)

            # length of the element = norm(p1-p2)
            ll = np.linalg.norm(p1-p2)

            # need to interpolate p from p1,p2 at each integration point, pass it to
            # global_1d_shape_der, get global derivatives and compare with those calculated from jaco1d

            for tt in zip(itertools.repeat([p1,p1]),der):
                # AssertionError must be raised when length of element is very small ( <1e-12 )
                self.assertRaises(AssertionError,jaco1d,*tt)

    def test_interp(self):
        # test 1d interpolation with scalars, vectors and matrices
        
        # 1d - 3 integration points
        gg   = gauss1d(3)
        *ss, = map(shape1d,gg.pts)
        shp  = [s.shape for s in ss ]


        d1 = np.asarray((1,));    d2 = np.asarray((1,))
        actout = interp_parent([d1,d2],shp[0])
        
            



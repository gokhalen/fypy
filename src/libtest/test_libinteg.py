import unittest,numpy as np,math,functools,itertools
from typing import Callable,Any,Union
from ..libinteg import *
from ..libshape import *
from .test import *


class TestLibInteg(TestFyPy):

    # methods to check integration points and integration routines
    @staticmethod
    def func1_N1(gausspt,shp,jac,data):
        # retuns N1 in the parent domain
        # gausspt: gauss point at which function evaluation has to be done
        # shp: shape functions and their derivatives at the gauss point
        return shp.shape[0]
    @staticmethod
    def func1_N2(gausspt,shp,jac,data):
        # retuns N2 in the parent domain
        # gausspt: gauss point at which function evaluation has to be done
        # shp: shape functions and their derivatives at the gauss point
        return shp.shape[1]

    @staticmethod
    def func1_N1N2(gausspt,shp,jac,data):
        # retuns N1N2 in the parent domain
        # gausspt: gauss point at which function evaluation has to be done
        # shp: shape functions and their derivatives at the gauss point

        # in 1D this is a function of degree 2, will require at least 2 gauss pts to integrate
        return shp.shape[0]*shp.shape[1]
    
    @staticmethod
    def func1_1(gausspt,shp,jac,data):
        return 1

    @staticmethod
    def func1_N1xN2x(gausspt,shp,jac,data):
        # for 1D, the derivatives of the shape functions are constants (-0.5,0.5)
        return shp.der[0][0]*shp.der[1][0]

    @staticmethod
    def func1_N1x(gausspt,shp,jac,data):
        return shp.der[0][0]

    @staticmethod
    def func1_N2x(gausspt,shp,jac,data):
        return shp.der[1][0]

    @staticmethod
    def func1_v_N1_N2(gausspt,shp,jac,data):
        # test vector integration the _v_ stands for vector
        return np.asarray((shp.shape[0],shp.shape[1]))

    @staticmethod
    def func1_m_1(gausspt,shp,jac,data):
        # _m_ stands for matrix
        # return [[ N1,N2],[N2,N1]]
        a00 = shp.shape[0]; a01 = shp.shape[1]
        a10 = shp.shape[1]; a11 = shp.shape[0]
        
        return np.asarray( ((a00,a01),(a10,a11)) )

    @staticmethod
    def func1_m_2(gausspt,shp,jac,data):
        # quadratic matrix function - needs 2 pt integration
        # return [[ N1*N1,N1*N21],[N2*N1,N2*N2]]
    
        a00 = shp.shape[0]*shp.shape[0]; a01 = shp.shape[0]*shp.shape[1]
        a10 = shp.shape[1]*shp.shape[0]; a11 = shp.shape[1]*shp.shape[1]

        return np.asarray( ((a00,a01),(a10,a11)) )
    
    @staticmethod
    def func2_N1(gausspt,shp,jac,data):
        return shp.shape[0]

    @staticmethod
    def func2_N2(gausspt,shp,jac,data):
        return shp.shape[1]

    @staticmethod
    def func2_N3(gausspt,shp,jac,data):
        return shp.shape[2]

    @staticmethod
    def func2_N4(gausspt,shp,jac,data):
        return shp.shape[3]

    @staticmethod
    def func2_N1pN2pN3pN4(gausspt,shp,jac,data):
        return (shp.shape[0] + shp.shape[1] + shp.shape[2] + shp.shape[3])

    @staticmethod
    def func2_N1N3(gausspt,shp,jac,data):
        return (shp.shape[0]*shp.shape[2])

    @staticmethod
    def func2_N1N2(gausspt,shp,jac,data):
        return (shp.shape[0]*shp.shape[1])

    @staticmethod
    def func2_N1N2N3(gausspt,shp,jac,data):
        # third degree function in each direction, can be integrated using 2pt rule
        return (shp.shape[0]*shp.shape[1]*shp.shape[2])

    @staticmethod
    def func2_v_N1x_N2y_N3x_N4y(gausspt,shp,jac,data):
        # 1pt integration
        n1x = shp.der[0][0]
        n2y = shp.der[1][1]
        n3x = shp.der[2][0]
        n4y = shp.der[3][1]
        return np.asarray((n1x,n2y,n3x,n4y))
    
    def test_integration_consistency(self):
        # consistency tests between gaussnd and gauss1d,gauss2d,gauss3d routines
        # gaussnd uses gauss1d, so results should be exactly equal.
        
        ngauss,npoint=3,10
        
        #print(f'Comparing consisteny of gaussnd and gauss1d,gauss2d,gauss3d for {npoint} integration points')
        #for idime,ipoint in itertools.product(range(1,ngauss+1),range(1,npoint+1)):
        #    self.compare_gaussnd(idime,ipoint)

        datamsg  = ['Integration points ', 'Weights ' ]

        for idime in range(1,4):
            fargs    = tuple([(i,) for i in range(1,npoint+1)])
            frefargs = tuple([(idime,i) for i in range(1,npoint+1)])
            ftest    = eval(f'gauss{idime}d')
            self.compare_test_func(ftest=ftest,fargs=fargs,fref=gaussnd,frefargs=frefargs,datamsg=datamsg,optmsg=f'Testing gauss{idime}d consistency ')
            
    def test_compare_integration_1d_hughes(self):
        # 1 point integration - weights from Hughes' linear
        # these tests will not be exactly equal because we are computing them two different ways
        # so use np.allclose
        npoints=1
        # ptsh = points hughes, wtsh = weights hughes
        # p.g. 141
        ptsh=np.asarray((0.0,))
        ptsh=ptsh[:,np.newaxis]  # need to make points an array of arrays: i.e. a matrix
        wtsh=np.asarray((2.0,))

        #self.compare_gaussnd(ndime=1,npoints=1,ptn=ptsh,wtn=wtsh,data_supplied=True)
        datamsg  = ['Points ', 'Weights ']
        truedata = [Integ(pts=ptsh,wts=wtsh)]
        optmsg   = 'Testing Gauss1d 1 point against precomputed data from TJRH ' 
        # sanity tests: must fail
        # truedata[0].pts[0]=11
        # truedata[0].wts[0] = 13
        self.compare_test_data(ftest=gauss1d, fargs=((npoints,),),truedata=truedata,datamsg=datamsg,optmsg=optmsg)

        # 2 point integration - weights from Hughes' linear
        # p.g. 141
        npoints=2
        ptsh=np.asarray((-1/math.sqrt(3.0),+1/math.sqrt(3.0)))
        ptsh=ptsh[:,np.newaxis]

        wtsh=np.asarray((1.0,1.0))
        
        #self.compare_gaussnd(ndime=1,npoints=2,ptn=ptsh,wtn=wtsh,data_supplied=True)
        truedata = [Integ(pts=ptsh,wts=wtsh)]
        optmsg   = 'Testing Gauss1d 2 point against precomputed data from TJRH '
        # Sanity tests must fail
        # truedata[0].pts[1] += 102
        # truedata[0].wts[0] +=32
        self.compare_test_data(ftest=gauss1d, fargs=((npoints,),),truedata=truedata,datamsg=datamsg,optmsg=optmsg)

        # p.g. 142
        npoints=3
        p1 = math.sqrt(3.0/5.0)
        ptsh=np.asarray((-p1,0.0,p1))
        ptsh=ptsh[:,np.newaxis]
        wtsh=np.asarray((5.0/9.0,8.0/9.0,5.0/9.0))

        #self.compare_gaussnd(ndime=1,npoints=3,ptn=ptsh,wtn=wtsh,data_supplied=True)
        truedata = [Integ(pts=ptsh,wts=wtsh)]
        optmsg   = 'Testing Gauss1d 3 point against precomputed data from TJRH '
        # Sanity tests must fail
        # truedata[0].pts[2] += 102
        # truedata[0].wts[0] +=32
        self.compare_test_data(ftest=gauss1d, fargs=((npoints,),),truedata=truedata,datamsg=datamsg,optmsg=optmsg)
        
    def test_compare_integration_2d_hughes(self):
        # Hughes p.g. 144 e.g. 6
        npoints = 1
        ptsh = np.asarray((0.0,0.0))
        ptsh = ptsh[np.newaxis,:]
        wtsh = np.asarray((4.0,))
        #self.compare_gaussnd(ndime=2,npoints=1,ptn=ptsh,wtn=wtsh,data_supplied=True)
        truedata = [Integ(pts=ptsh,wts=wtsh)]
        optmsg   = 'Testing Gauss2d 1 point against precomputed data from TJRH '
        datamsg  = ['Points ', 'Weights ']
        # Sanity tests must fail
        # truedata[0].pts[0][1] += 102
        # truedata[0].wts[0] +=32
        self.compare_test_data(ftest=gauss2d, fargs=((npoints,),),truedata=truedata,datamsg=datamsg,optmsg=optmsg)

        # p.g. 145
        npoints = 2
        pt      = math.sqrt(1.0/3.0)
        ptsh    = np.asarray(((-pt,-pt),(-pt,pt),(pt,-pt),(pt,pt)))
        wtsh    = np.asarray((1.0,1.0,1.0,1.0))
        #self.compare_gaussnd(ndime=2,npoints=2,ptn=ptsh,wtn=wtsh,data_supplied=True)
        
        truedata = [Integ(pts=ptsh,wts=wtsh)]
        optmsg   = 'Testing Gauss2d 2 point against precomputed data from TJRH '
        datamsg  = ['Points ', 'Weights ']
        # Sanity tests must fail
        # truedata[0].pts[3][1] += 102
        # truedata[0].wts[3] +=32        
        # from gauss2d.cpp in hyser, can also be derived from the 3pt rule above
        self.compare_test_data(ftest=gauss2d, fargs=((npoints,),),truedata=truedata,datamsg=datamsg,optmsg=optmsg)
        
        npoints = 3
        pt=math.sqrt(3.0/5.0)
        ptsh=np.asarray(((-pt,-pt),(-pt,0.0),(-pt,pt),(0.0,-pt),(0.0,0.0),(0.0,pt),(pt,-pt),(pt,0.0),(pt,pt)))
        w1,w2,w3 = 5.0/9.0,8.0/9.0,5.0/9.0
        wtsh=np.asarray((w1*w1, w1*w2, w1*w3, w2*w1, w2*w2, w2*w3, w3*w1, w3*w2, w3*w3 ))
        #self.compare_gaussnd(ndime=2,npoints=3,ptn=ptsh,wtn=wtsh,data_supplied=True)

        truedata = [Integ(pts=ptsh,wts=wtsh)]
        optmsg   = 'Testing Gauss2d 3 point against precomputed data from TJRH '
        datamsg  = ['Points ', 'Weights ']
        # Sanity tests must fail
        # truedata[0].pts[8][0] += 102
        # truedata[0].wts[6] +=32     
        self.compare_test_data(ftest=gauss2d, fargs=((npoints,),),truedata=truedata,datamsg=datamsg,optmsg=optmsg)
        
    def test_compare_integration_3d_hughes(self):
        
        npoints = 1
        ptsh = np.asarray(((0.0,0.0,0.0),))
        wtsh = np.asarray((8.0,))
        #self.compare_gaussnd(ndime=3,npoints=1,ptn=ptsh,wtn=wtsh,data_supplied=True)

        truedata = [Integ(pts=ptsh,wts=wtsh)]
        optmsg   = 'Testing Gauss2d 3 point against precomputed data from TJRH '
        datamsg  = ['Points ', 'Weights ']
        
        # Sanity tests must fail
        # truedata[0].pts[0][0] += 102
        # truedata[0].wts[0]    +=32
        
        self.compare_test_data(ftest=gauss3d, fargs=((npoints,),),truedata=truedata,datamsg=datamsg,optmsg=optmsg)

        # from gauss3d.cpp in hyser
        npoints = 2
        pt = math.sqrt(1/3.0)
        ptsh = np.asarray(((-pt,-pt,-pt),(-pt,-pt,pt),(-pt,pt,-pt),(-pt,pt,pt),(pt,-pt,-pt),(pt,-pt,pt),(pt,pt,-pt),(pt,pt,pt)))
        wtsh = np.asarray((1.0,)*8)
        #self.compare_gaussnd(ndime=3,npoints=2,ptn=ptsh,wtn=wtsh,data_supplied=True)
        
        truedata = [Integ(pts=ptsh,wts=wtsh)]
        optmsg   = 'Testing Gauss2d 3 point against precomputed data from TJRH '
        datamsg  = ['Points ', 'Weights ']

        # Sanity tests must fail
        # truedata[0].pts[7][1] += 102
        # truedata[0].wts[3]    +=32
        
        self.compare_test_data(ftest=gauss3d, fargs=((npoints,),),truedata=truedata,datamsg=datamsg,optmsg=optmsg)

    def test_parent_integration_1d(self):
        # 0 is the minimum number of points needed to integrate N_a in 1D
        # n gauss points integrate function of order (2n) i.e. (degree 2n-1) exactly
        
        funclistlin = [self.func1_1,self.func1_N1,self.func1_N2,self.func1_N1xN2x,self.func1_N1x,self.func1_N2x,self.func1_v_N1_N2,self.func1_m_1]
        expvallin   = [ 2, 1, 1, -0.5, -1, 1, np.asarray((1,1)), np.asarray(( (1,1),(1,1) ))   ]

        funclistquad = [self.func1_N1N2, self.func1_m_2]
        expvalquad   = [1/3,            np.asarray(( (2.0/3.0,1/3),(1/3,2.0/3.0) ))]

        for ipoint in range(1,10):
            gg    = gauss1d(ipoint)
            *ss,  = map(shape1d,gg.pts)
            data  = [None]*ipoint
            *jac, = map(Jaco,itertools.repeat(None),itertools.repeat(1),itertools.repeat(None,ipoint)) 

            # integral value
            # test linear functions
            for ftest,fval in zip(funclistlin,expvallin):
                intval  = integrate_parent(ftest,gg,ss,data,jac)
                boolcmp = npclose(intval,fval) 
                self.assertTrue(boolcmp,msg=f'test_integration_1d linear failed for {ipoint=} func={ftest.__name__}')

            # test quadratic functions
            if (ipoint >= 2):
                for ftest,fval in zip(funclistquad,expvalquad):
                    intval  = integrate_parent(ftest,gg,ss,data,jac)
                    boolcmp = npclose(intval,fval) 
                    self.assertTrue(boolcmp,msg=f'test_integration_1d quad failed for {ipoint=} func={ftest.__name__}')

    def test_parent_integration_2d(self):

        # TEST SHAPE FUNCTION DERIVATIVE INTEGRATION
        
        funclistlin =[]; expvallin  = []
        funclistquad=[]; expvalquad = []

        funclistlin.append(self.func2_N1);                  expvallin.append(1)
        funclistlin.append(self.func2_N2);                  expvallin.append(1)
        funclistlin.append(self.func2_N3);                  expvallin.append(1)
        funclistlin.append(self.func2_N4);                  expvallin.append(1)
        funclistlin.append(self.func2_N1pN2pN3pN4);         expvallin.append(4)
        
        _exp = np.asarray( (-1.0,-1.0,1.0,1.0) )
        funclistlin.append(self.func2_v_N1x_N2y_N3x_N4y);   expvallin.append(_exp)

        funclistquad.append(self.func2_N1N3);               expvalquad.append(1.0/9)
        funclistquad.append(self.func2_N1N2);               expvalquad.append(2.0/9)
        funclistquad.append(self.func2_N1N2N3);             expvalquad.append(1.0/36)

        for ipoint in range(1,10):
            gg    = gauss2d(ipoint)
            *ss,  = map(shape2d,gg.pts)
            data  = [None]*ipoint*ipoint
            *jac, = map(Jaco,itertools.repeat(None),itertools.repeat(1),itertools.repeat(None,ipoint*ipoint)) 

            # test functions which need one point (in each direction) integration
            for ftest,fval in zip(funclistlin,expvallin):
                intval  = integrate_parent(ftest,gg,ss,data,jac)
                boolcmp = npclose(intval,fval)
                self.assertTrue(boolcmp,msg=f'test_integration_2d linear failed for {ipoint=} func={ftest.__name__} {intval=} {fval=}')

            # test functions which need two point (in each direction) integration
            if (ipoint >= 2):
                for ftest,fval in zip(funclistquad,expvalquad):
                    intval  = integrate_parent(ftest,gg,ss,data,jac)
                    boolcmp = npclose(intval,fval)
                    self.assertTrue(boolcmp,msg=f'test_integration_2d quad failed for {ipoint=} func={ftest.__name__} {intval=} {fval=}')

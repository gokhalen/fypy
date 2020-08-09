import unittest,numpy as np,math,functools,itertools
from typing import Callable,Any,Union
from ..libinteg.gausslegendre import *
from ..libinteg.integrate import integrate_parent
from ..libshape.shape import *
from .test import *


class TestLibInteg(TestFyPy):

    # methods to check integration points and integration routines
    @staticmethod
    def func_N1(gausspt,shp,data):
        # retuns N1 in the parent domain
        # gausspt: gauss point at which function evaluation has to be done
        # shp: shape functions and their derivatives at the gauss point
        return shp.shape[0]
    @staticmethod
    def func_N2(gausspt,shp,data):
        # retuns N2 in the parent domain
        # gausspt: gauss point at which function evaluation has to be done
        # shp: shape functions and their derivatives at the gauss point
        return shp.shape[1]

    @staticmethod
    def func_N1N2(gausspt,shp,data):
        # retuns N1N2 in the parent domain
        # gausspt: gauss point at which function evaluation has to be done
        # shp: shape functions and their derivatives at the gauss point

        # in 1D this is a function of degree 2, will require at least 2 gauss pts to integrate
        return shp.shape[0]*shp.shape[1]
    
    @staticmethod
    def func_1(gaussp1,shp,data):
        return 1
    
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
        # what is the minimum number of points needed to integrate N_a in 1D
        funclistlin = [self.func_1,self.func_N1,self.func_N2]
        expvallin   = [ 2, 1, 1 ]

        funclistquad = [self.func_N1N2]
        expvalquad   = [1/3]
        
        for ipoint in range(1,10):
            gg    = gauss1d(ipoint)
            *ss,  = map(shape1d,gg.pts)
            data  = [None]*ipoint
            wtjac = gg.wts

            # integral value
            # test linear functions
            for ftest,fval in zip(funclistlin,expvallin):
                intval = integrate_parent(ftest,gg.pts,ss,data,wtjac)
                self.assertAlmostEqual(fval,intval,places=closeplaces,msg=f'test_integration_1d failed for {ipoint=} func={ftest.__name__}')

            # test quadratic functions
            if (ipoint >= 2):
                for ftest,fval in zip(funclistquad,expvalquad):
                    intval = integrate_parent(ftest,gg.pts,ss,data,wtjac)
                    self.assertAlmostEqual(fval,intval,places=closeplaces,msg=f'test_integration_1d failed for {ipoint=} func={ftest.__name__}')
                

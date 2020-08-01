import unittest,numpy as np,math,functools,itertools
from typing import Callable,Any,Union
from ..libinteg.gausslegendre import gauss1d,gauss2d,gauss3d,gaussnd
from .test import *


class TestLibInteg(TestFyPy):
    # consistency tests between gaussnd and gauss1d,gauss2d,gauss3d routines
    # gaussnd uses gauss1d, so results should be exactly equal.
    
    def compare_gaussnd(self,ndime:int,npoints:int,ptn=None,wtn=None,data_supplied=False)->Union[None,AssertionError]:
        # this method returns either None or an raises an AssertionError.
        # I've type hinted it to return either, though type hinting for exceptions seems verboten
        # fgauss: either gauss1d,gauss2d,gauss3d that can be compared against gaussnd
        # nfunc: which gauss function to be called (1,2 or 3)

        msgpt, msgwt = '',''
        pt,wt        = eval(f'gauss{ndime}d(npoints)')
        
        if ( not data_supplied ):
            ptn,wtn = gaussnd(ndime,npoints)
            msg     = f'gauss{ndime}d does not match gaussnd for {npoints} integration points'

        if ( data_supplied ) :
            msg = f"gauss{ndime}d does not match Hughes' data for {npoints} integration points"

        # To make sure the test is working, set arrays to wrong value
        # if ( (ndime==2) and (npoints==3) ):
        #    pt[3][1] += 1
        #    wt[4]    += 1
        
        boolpt,boolwt  = map(npclose,(ptn,wtn),(pt,wt))

        if (not boolpt):
            # call function to determine which entry is off
            # if really wants to get fancy, we can decorate get_mismatch with make_mismatch_message
            idx,aa,bb = get_mismatch(ptn,pt,closetol=closetol)
            msgpt     = make_mismatch_message(idx,aa,bb)

        if (not boolwt):
            # call function to determine which entry is off
            idx,aa,bb  = get_mismatch(wtn,wt,closetol=closetol)
            msgwt      = make_mismatch_message(idx,aa,bb)
            
        self.assertTrue(boolpt,msg=msg+msgpt)
        self.assertTrue(boolwt,msg=msg+msgwt)

    def test_integration_consistency(self):
        ngauss,npoint=3,10
        print(f'Comparing consisteny of gaussnd and gauss1d,gauss2d,gauss3d for {npoint} integration points')
        for idime,ipoint in itertools.product(range(1,ngauss+1),range(1,npoint+1)):
            self.compare_gaussnd(idime,ipoint)

        datamsg  = ['Integration points ', 'Weights ' ]

        for idime in range(1,4):
            fargs    = tuple([(i,) for i in range(1,npoint+1)])
            frefargs = tuple([(idime,i) for i in range(1,npoint+1)])
            ftest    = eval(f'gauss{idime}d')
            self.compare_test_data(ftest,fargs,gaussnd,frefargs,None,datamsg,data_supplied=False,optmsg=f'Testing gauss{idime}d consistency ')
            
    def compare_hughes(self,npoints,ftest:Callable,ptsh,wtsh):
        '''
        npoints: number of points to use for integration in each direction
        ftest:   a function which is to be tested against the data from
                 Hughes' book
        ptsh:  integration points from Hughes' book
        wtsh:  weights from Hughes' boook
        '''
        msgpts = f"{ftest.__name__} {npoints=} pts does not match Hughes' linear"
        msgwts = f"{ftest.__name__} {npoints=} wts does not match Hughes' linear"

        (pts,wts)=map(np.asarray,ftest(npoints))
        (wtclose,ptclose)=map(npclose,(pts,wts),(ptsh,wtsh))
        
        self.assertTrue(wtclose,msg=msgwts)
        self.assertTrue(ptclose,msg=msgpts)
        

    
    def test_integration_1d_hughes(self):
        # 1 point integration - weights from Hughes' linear
        # these tests will not be exactly equal because we are computing them two different ways
        # so use np.allclose
        npoints=1
        # ptsh = points hughes, wtsh = weights hughes
        # p.g. 141
        ptsh=np.asarray((0.0,))
        ptsh=ptsh[:,np.newaxis]  # need to make points an array of arrays: i.e. a matrix
        wtsh=np.asarray((2.0,))

        self.compare_gaussnd(ndime=1,npoints=1,ptn=ptsh,wtn=wtsh,data_supplied=True)
        

        # 2 point integration - weights from Hughes' linear
        # p.g. 141
        npoints=2
        ptsh=np.asarray((-1/math.sqrt(3.0),+1/math.sqrt(3.0)))
        ptsh=ptsh[:,np.newaxis]

        wtsh=np.asarray((1.0,1.0))
        
        self.compare_gaussnd(ndime=1,npoints=2,ptn=ptsh,wtn=wtsh,data_supplied=True)
        
        # p.g. 142
        npoints=3
        p1 = math.sqrt(3.0/5.0)
        ptsh=np.asarray((-p1,0.0,p1))
        ptsh=ptsh[:,np.newaxis]

        wtsh=np.asarray((5.0/9.0,8.0/9.0,5.0/9.0))

        self.compare_gaussnd(ndime=1,npoints=3,ptn=ptsh,wtn=wtsh,data_supplied=True)
        
    def test_integration_2d_hughes(self):
        # Hughes p.g. 144 e.g. 6
        npoints = 1
        ptsh = np.asarray((0.0,0.0))
        ptsh = ptsh[:,np.newaxis]
        wtsh = np.asarray((4.0,))
        self.compare_gaussnd(ndime=2,npoints=1,ptn=ptsh,wtn=wtsh,data_supplied=True)
        
        # p.g. 145
        npoints = 2
        pt      = math.sqrt(1.0/3.0)
        ptsh    = np.asarray(((-pt,-pt),(-pt,pt),(pt,-pt),(pt,pt)))
        wtsh    = np.asarray((1.0,1.0,1.0,1.0))
        self.compare_gaussnd(ndime=2,npoints=2,ptn=ptsh,wtn=wtsh,data_supplied=True)
        
        # from gauss2d.cpp in hyser, can also be derived from the 3pt rule above
        
        npoints = 3
        pt=math.sqrt(3.0/5.0)
        ptsh=np.asarray(((-pt,-pt),(-pt,0.0),(-pt,pt),(0.0,-pt),(0.0,0.0),(0.0,pt),(pt,-pt),(pt,0.0),(pt,pt)))
        w1,w2,w3 = 5.0/9.0,8.0/9.0,5.0/9.0
        wtsh=np.asarray((w1*w1, w1*w2, w1*w3, w2*w1, w2*w2, w2*w3, w3*w1, w3*w2, w3*w3 ))
        self.compare_gaussnd(ndime=2,npoints=3,ptn=ptsh,wtn=wtsh,data_supplied=True)
        
    def test_integration_3d_hughes(self):
        
        npoints = 1
        ptsh = np.asarray(((0.0,0.0,0.0),))
        wtsh = np.asarray((8.0))
        self.compare_gaussnd(ndime=3,npoints=1,ptn=ptsh,wtn=wtsh,data_supplied=True)
        
        # from gauss3d.cpp in hyser
        npoints = 2
        pt = math.sqrt(1/3.0)
        ptsh = np.asarray(((-pt,-pt,-pt),(-pt,-pt,pt),(-pt,pt,-pt),(-pt,pt,pt),(pt,-pt,-pt),(pt,-pt,pt),(pt,pt,-pt),(pt,pt,pt)))
        wtsh = np.asarray((1.0,)*8)
        self.compare_gaussnd(ndime=3,npoints=2,ptn=ptsh,wtn=wtsh,data_supplied=True)
        
        


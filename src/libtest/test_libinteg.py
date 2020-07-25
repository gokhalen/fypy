import unittest
import numpy as np
import math
import functools

from typing import Callable
from ..libinteg.gausslegendre import gauss1d,gauss2d,gauss3d,gaussnd

closetol = 1e-12

npclose=functools.partial(np.allclose,atol=closetol)

class TestLibInteg(unittest.TestCase):
    # consistency tests between gaussnd and gauss1d,gauss2d,gauss3d routines
    # gaussnd uses gauss1d, so results should be exactly equal.
    def test_integration_consistency(self):
        for npoints in range(1,10):
            self.assertEqual(gauss1d(npoints),gaussnd(ndim=1,npoints=npoints),msg='gauss1d consistency failure')
            self.assertEqual(gauss2d(npoints),gaussnd(ndim=2,npoints=npoints),msg='gauss2d consistency failure')
            self.assertEqual(gauss3d(npoints),gaussnd(ndim=3,npoints=npoints),msg='gauss3d consistency failure')
    
    
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
        
        pass
    
    def test_integration_1d_hughes(self):
        # 1 point integration - weights from Hughes' linear
        # these tests will not be exactly equal because we are computing them two different ways
        # so use np.allclose
        npoints=1
        # ptsh = points hughes, wtsh = weights hughes
        # p.g. 141
        ptsh=np.asarray((0.0,))
        wtsh=np.asarray((2.0,))
        
        self.compare_hughes(npoints,gauss1d,ptsh,wtsh)

        # 2 point integration - weights from Hughes' linear
        # p.g. 141
        npoints=2
        ptsh=np.asarray((-1/math.sqrt(3.0),+1/math.sqrt(3.0)))
        wtsh=np.asarray((1.0,1.0))
        
        self.compare_hughes(npoints,gauss1d,ptsh,wtsh)
        
        # p.g. 142
        npoints=3
        p1 = math.sqrt(3.0/5.0)
        ptsh=np.asarray((-p1,0.0,p1))
        wtsh=np.asarray((5.0/9.0,8.0/9.0,5.0/9.0))
        
        self.compare_hughes(npoints,gauss1d,ptsh,wtsh)
        
    def test_integration_2d_hughes(self):
        # p.g. 144 e.g. 6
        npoints = 1
        ptsh = np.asarray((0.0,0.0))
        wtsh = np.asarray((4.0))
        self.compare_hughes(npoints,gauss2d,ptsh,wtsh)
        
        # p.g. 145
        npoints = 2
        pt      = math.sqrt(1.0/3.0)
        ptsh    = np.asarray(((-pt,-pt),(-pt,pt),(pt,-pt),(pt,pt))) 
        wtsh    = np.asarray((1.0,1.0,1.0,1.0))
        self.compare_hughes(npoints,gauss2d,ptsh,wtsh)

        # gauss2d.cpp in hyser, can also be derived from the 3pt rule above
        
        npoints = 3
        pt=math.sqrt(3.0/5.0)
        ptsh=np.asarray(((-pt,-pt),(-pt,0.0),(-pt,pt),(0.0,-pt),(0.0,0.0),(0.0,pt),(pt,-pt),(pt,0.0),(pt,pt)))
        w1,w2,w3 = 5.0/9.0,8.0/9.0,5.0/9.0
        wtsh=np.asarray((w1*w1, w1*w2, w1*w3, w2*w1, w2*w2, w2*w3, w3*w1, w3*w2, w3*w3 ))
        self.compare_hughes(npoints,gauss2d,ptsh,wtsh)
        
    def test_integration_3d_hughes(self):
        
        npoints = 1
        ptsh = np.asarray((0.0,0.0,0.0))
        wtsh = np.asarray((8.0))
        self.compare_hughes(npoints,gauss3d,ptsh,wtsh)
        
        # from gauss3d.cpp in hyser
        npoints = 2
        pt = math.sqrt(1/3.0)
        ptsh = np.asarray(((-pt,-pt,-pt),(-pt,-pt,pt),(-pt,pt,-pt),(-pt,pt,pt),(pt,-pt,-pt),(pt,-pt,pt),(pt,pt,-pt),(pt,pt,pt)))
        wtsh = np.asarray((1.0,)*8)
        self.compare_hughes(npoints,gauss3d,ptsh,wtsh)
        
        


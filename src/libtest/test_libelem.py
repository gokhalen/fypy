import numpy as np

from ..libelem.linelas1d import *
from ..libelem.linelas2d import *
from .test import *


class TestLibElem(TestFyPy):
    
    def test_linelas1d(self):

        # the problem we're solving in (d/dx)(k(x) du/dx) = l
        # on (2.3,5.15) k(2.3) = 2.15 and k(5.15) = 7.25

        # for the global problem, once we get there
        # u(2.3) = 1.7 and traction at (5.15) = 2.1
        # body force = 2x^2
        
        
        elas1d = LinElas1D(ninteg=3,gdofn=10)
        coord = np.zeros(6,dtype='float64').reshape(2,3)
        
        coord[0][0] =  2.3;  coord[0][1]=0.0; coord[0][2] = 0.0;
        coord[1][0] =  5.15; coord[1][1]=0.0; coord[1][2] = 0.0;

        # set material properties and body force
        prop   = np.arange(2,dtype='float64').reshape(2,1)
        bf     = np.arange(2,dtype='float64').reshape(2,1)
        pforce = np.arange(2,dtype='float64').reshape(2,1)
        dirich = np.arange(2,dtype='float64').reshape(2,1)
        trac   = np.arange(2,dtype='float64').reshape(2,1)

        isbc   = np.zeros(2).reshape(2,1)
        ideqn  = np.zeros(2).reshape(2,1)
        
        prop[0][0]   = 2.15;  prop[1][0]   = 7.25
        bf[0][0]     = 11.2;  bf[1][0]     = 13.7
        pforce[0][0] = 23.1;  pforce[1][0] = 17.2
        dirich[0][0] = 2.718; dirich[1][0] = 3.14
        trac[0][0]   = 3.1;   trac[1][0]   = 6.7
        
        isbc[0][0]   = 0;     isbc[1][0]   = 1
        ideqn[0][0]  = 1;     ideqn[1][0]  = 4;
        

        elas1d.setdata(coord=coord,prop=prop,bf=bf,pforce=pforce,dirich=dirich,trac=trac,ideqn=ideqn,isbc=isbc)
        elas1d.compute()

        # check stiffness matrix
        kk = 1.6491228070175434
        expstiff = np.asarray( ( (kk,-kk), (-kk,kk) ))
        msg='In test_linelas1d stiffness matrix comparison for linelas1d fails '
        self.compare_iterables(elas1d.estiff,expstiff,msg=msg,desc='')

        # check rhs body force
        expbf = np.asarray(( 17.1475, 18.335 ))
        msg='In test_linelas1d body force rhs comparison for linelas1d fails '
        self.compare_iterables(elas1d.erhsbf,expbf,msg=msg,desc='')

        # check the point force
        exppf = np.asarray(( 23.1,17.2))
        msg = 'In test_linelas1d point force rhs comparison for linelas1d fails'
        self.compare_iterables(elas1d.erhspf,exppf,msg=msg,desc='')

        # check dirichlet force
        expdir = -expstiff@np.asarray((dirich[0][0],dirich[1][0]))
        msg = 'In test_linelas1d dirichlet force rhs comparison for linelas1d fails'
        self.compare_iterables(elas1d.erhsdir,expdir,msg=msg,desc='')

        # check traction force - continuum elements do not implement traction
        exptrac = np.asarray((0.0,0.0))
        msg = 'In test_linelas1d traction force rhs comparison for linelas1d fails'
        self.compare_iterables(elas1d.erhstrac,exptrac,msg=msg,desc='')
                                      


        


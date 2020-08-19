import numpy as np

from ..libelem.linelas1d import *
from ..libelem.linelas2d import *
from .test import *


class TestLibElem(TestFyPy):
    
    def test_linelas1d(self):

        # the problem we're solving in (d/dx)(k(x) du/dx) = l
        # on (2.3,5.15) k(2.3) = 2.15 and k(5.15) = 7.25
        
        
        elas1d = LinElas1D(ninteg=3,gdofn=10)
        coord = np.zeros(6,dtype='float64').reshape(2,3)
        
        coord[0][0] =  2.3;  coord[0][1]=0.0; coord[0][2] = 0.0;
        coord[1][0] =  5.15; coord[1][1]=0.0; coord[1][2] = 0.0;

        # set material properties and body force
        prop   = np.arange(2,dtype='float64').reshape(2,1)
        bf     = np.arange(2,dtype='float64').reshape(2,1)
        
        prop[0][0] = 2.15; prop[1][0] = 7.25
        bf[0][0]   = 11.2; bf[1][0]   = 13.7
        

        
        
        elas1d.coord = coord
        elas1d.getjaco()
        elas1d.prop = prop
        elas1d.bf   = bf
        elas1d.interp()
        elas1d.compute_stiffness()
        elas1d.compute_rhs()

        # check stiffness matrix
        kk = 1.6491228070175434
        expstiff = np.asarray( ( (kk,-kk), (-kk,kk) ))
        
        msg='In test_linelas1d stiffness matrix comparison for linelas1d fails '
        self.compare_iterables(elas1d.estiff,expstiff,msg=msg,desc='')

        # check rhs body force
        expbf = np.asarray(( 17.1475, 18.335 ))

        msg='In test_linelas1d body force rhs comparison for linelas1d fails '
        self.compare_iterables(elas1d.erhsbf,expbf,msg=msg,desc='')

        


        


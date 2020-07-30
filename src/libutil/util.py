# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 15:22:56 2020

@author: Nachiket Gokhale

defines basic classes Point,ParentPoint

"""

from typing import Tuple
import numpy as np
import math

class Point(object):
    '''
    A basic Point class in n-dimensional space
    '''
    def __init__(self,coord:Tuple[float,...]):
        self._x = np.array(coord)
        
    def __getitem__(self,i:int):
        return self._x[i]
    
    def __setitem__(self,i:int,value:float):
        self._x[i] = value

    def __len__(self):
        return len(self._x)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self,value):
        self._x = value

#   https://stackoverflow.com/a/33533514/13560598
#   Also, see notes

    def distance(self,P1:'Point')->float:
        return math.sqrt(sum((self.x - P1.x)**2.0)) 

        
class ParentPoint(Point):
    '''
    A point in the parent domain
    Checks to see if the coords lie in [-1,1]\cross[-1,-1]\cross[-1,-1]
    '''
    
    def __init__(self,coord:Tuple[float,float,float]):
        
        assert (len(coord)==3),'ParentPoints must be 3-dimensional'
        
        for i in range(3):
            assert (-1 <= coord[i] <= 1), f'{i}-th parent domain coordinate\
                                         does not lie in the interval [-1,1]' 
        super(ParentPoint,self).__init__(coord)
    
      
    # override Point's __setitem__ to check bounds before setting
    
    def __setitem__(self,i:int,value:float):
        assert (-1 <= value <= 1), f'trying to set {i}-th dimension \
                     to {value} which is outside the interval [-1,1]'
                     
        super(ParentPoint,self).__setitem__(i,value)
                     


def get_mismatch(aa,bb,closetol=1E-12):
   # if entries of aa and bb are off by an amount greater than closetol
   # then the indices and values of those entries are returned
   # if no entries are mismatched, empties are returned
   idx       = np.where(abs(aa-bb)>closetol)
   idxtuple  = tuple([ *zip(*idx)])
   valuesa   = aa[idx]
   valuesb   = bb[idx]
   return (idxtuple,valuesa,valuesb)

def make_mismatch_message(idxtuple,aa,bb):
    # makes a string showing the locations and values that do not match
    # to be used with: get_mismatch
    msg='Entries do not match: '
    for ii,(idx,va,vb) in enumerate(zip(idxtuple,aa,bb)):
         msg = msg + "at location " + str(idx) + " (" + str(aa[ii]) + " != " + str(bb[ii]) + ") "
    return msg       
    

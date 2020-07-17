# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 15:22:56 2020

@author: Nachiket Gokhale

defines basic classes Point,ParentPoint

"""

from typing import Tuple


class Point(object):
    '''
    A basic Point class in n-dimensional space
    '''
    def __init__(self,coord:Tuple[float,...]):
        self._x = list(coord)
        
    def __getitem__(self,i):
        return self._x[i]
    
    def __setitem__(self,i,value):
        self._x[i] = value

        
class ParentPoint(Point):
    '''
    A point in the parent domain
    Checks to see if the coords lie in [-1,1]\cross[-1,-1]\cross[-1,-1]
    '''
    def __init__(self,coord:Tuple[float,...]):
        assert (len(coord)==3),'ParentPoints must be 3-dimensional'
        for i in range(3):
            assert (-1 <= coord[i] <= 1), f'{i}-th parent domain coordinate\
                                           does not lie in the interval [-1,1]' 
        super(ParentPoint,self).__init__(coord)
    
    
    
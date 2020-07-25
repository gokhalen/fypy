# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 10:36:24 2020

@author: aa
"""


import numpy as np
import scipy.sparse

def getmatrix():
    
    rows=[0,1,2]    
    cols=[0,1,2]    
    data=[0,1,2]    
    N = 10
    mat = scipy.sparse.coo_matrix((data,(rows,cols)),shape=(N,N))
    return mat



 
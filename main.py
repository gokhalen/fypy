# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:31:21 2020

@author: aa
"""

import sys,os

from src.libshape import *
from src.libutil import *

if __name__ == '__main__':
    print('Executing fypy/main.py')

    output_prefix = ''
    if ( len(sys.argv) == 3):
        output_prefix = sys.argv[2]

    print(output_prefix)
    outfilename  = output_prefix + '/data.out'
    with open(outfilename,'w') as f:
        f.write('Dummy data')
    


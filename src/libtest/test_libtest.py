import numpy as np
from .test import *

class TestLibTest(TestFyPy):
    
    def test_compare_test_data(self):
        # generate a random matrix of random shape
        nrows=np.random.randint(1,1024)
        ncols=np.random.randint(1,1024)
        data=np.random.rand(nrows,ncols)

                                
        



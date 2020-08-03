import numpy as np
from .test import *

class TestLibTest(TestFyPy):

    @staticmethod
    def func1(xx):
        return (xx*xx + 2*xx + 3,)

    @staticmethod
    def func2(xx):
        return (3*xx + 5*xx**0.5,)
    
    def test_compare_test_data(self):
        # generate random data for testing purposes
        # Note: truedata has to yield iterables. compare_test_data iterates over those iterables, gets numpy arrays, and compares them.
        # Similarly, the functions to be tested have to yield iterables. compare_test_data iterates over those iterables, gets numpy arrays, and compares them.
        print(' Testing compare_test_data and compare_test_func ... ')
        
        ftestlist  = [ self.func1, self.func2 ]
        nrows      = np.random.randint(1,1024)
        ncols      = np.random.randint(1,1024)
        fargs1     = np.random.rand(nrows,ncols)
        fargs2     = np.random.rand(nrows,ncols)

        for ftest in ftestlist:
            print('in ftest loop')
            fargs      = [ (fargs1,),(fargs2,) ]
            truedata   = [ ftest(*args) for args in fargs]

            #sanity check must fail
            #rowwrng = np.random.randint(0,nrows)
            #colwrng = np.random.randint(0,ncols)
            #truedata[0][0][rowwrng][colwrng] +=1100
            #print('wrong row= ',rowwrng, 'wrong col= ',colwrng) 
            
            self.compare_test_data(ftest=ftest,fargs=fargs,truedata=truedata,datamsg=['fargs1'],optmsg='')
            # print(truedata[0][nrows-1][ncols-1],ftest(*fargs[0])[nrows-1][ncols-1])
            
            

                                
        


import numpy as np
from collections import namedtuple
from .test import *

class TestLibTest(TestFyPy):

    TestData = namedtuple('TestData','field1')
    
    @staticmethod
    def func1(xx):
        return TestLibTest.TestData(field1=(xx*xx + 2*xx + 3))

    @staticmethod
    def func2(xx):
        return TestLibTest.TestData(field1=(3*xx + 5*xx**0.5))

    @staticmethod
    def funcwrong1(xx,wrngrow,wrngcol):
        (wrongdata,) = TestLibTest.func1(xx)
        wrongdata[wrngrow,wrngcol] +=100
        return TestLibTest.TestData(field1=wrongdata)

    @staticmethod
    def funcwrong2(xx,wrngrow,wrngcol):
        (wrongdata,) = TestLibTest.func2(xx)
        wrongdata[wrngrow,wrngcol] +=100
        return TestLibTest.TestData(field1=wrongdata)

    def test_compare_test_data(self):
        # generate random data for testing purposes
        # Note: truedata has to yield iterables. compare_test_data iterates over those iterables, gets numpy arrays, and compares them.
        # Similarly, the functions to be tested have to yield iterables. compare_test_data iterates over those iterables, gets numpy arrays, and compares them.
        # print(' Testing compare_test_data and compare_test_func ... ')
        
        ftestlist  = [ self.func1, self.func2 ]
        fwronglist = [ self.funcwrong1, self.funcwrong2 ]
        
        nrows      = np.random.randint(1,1024)
        ncols      = np.random.randint(1,1024)
        fargs1     = np.random.rand(nrows,ncols)
        fargs2     = np.random.rand(nrows,ncols)

        for ftest,fwrong in zip(ftestlist,fwronglist):
            fargs      = [ (fargs1,),(fargs2,) ]
            truedata   = [ ftest(*args) for args in fargs]

            #sanity check must fail
            # rowwrng = np.random.randint(0,nrows)
            # colwrng = np.random.randint(0,ncols)
            # truedata[0][0][rowwrng][colwrng] +=1100
            # print('wrong row= ',rowwrng, 'wrong col= ',colwrng)
            # fwrongargs = [ (fargs1,rowwrng,colwrng),(fargs2,rowwrng,colwrng)]
            
            self.compare_test_data(ftest=ftest,fargs=fargs,truedata=truedata,datamsg=['data1'],optmsg='Testing compare_test_data ... ')
            # correct function test
            self.compare_test_func(ftest=ftest,fargs=fargs,fref=ftest,frefargs=fargs,datamsg=['data1'],optmsg='Testing compare_test_func ... ')
            #wrong function test
            #self.compare_test_func(ftest=ftest,fargs=fargs,fref=fwrong,frefargs=fwrongargs,datamsg=['data1'],optmsg='Testing compare_test_func ... ')
            
            
        

                                
        



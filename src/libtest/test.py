import functools,numpy as np,unittest,itertools
from typing import Callable,Tuple,Iterable
closetol=1e-12
npclose=functools.partial(np.allclose,atol=closetol)

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

class TestFyPy(unittest.TestCase):
   def compare_test_data(self,ftest:Callable,fargs:Iterable,fref:Callable,frefargs:Iterable,truedata:Iterable,datamsg:Iterable,data_supplied=False,optmsg=None):
      '''
      ftest    : Callable to be tested
      fargs    : Iterable yielding tuples which are used as 
      fref     : the reference function against which to compare
      frefargs : Iterable yielding the tuples which when unpacked are arguments to fref
      truedata : Iterable yielding reference data (named tuples) which is to be compared
      datamsg  : iterable yielding strings describing the data to be tested. Size is the size of the tuples yielded by truedata or by calling ftest
      data_supplied: If True, ftest is not called to generate reference data. If False, fref is called
      optmsg   : optional message
      '''

      outmsg=''
      if ( not data_supplied ):
         truedata=itertools.repeat(None)

      if (( not data_supplied ) and ( fref == None )):
         raise RuntimeError('TestFyPy: If data is not supplied, fref cannot be None')

      if ( data_supplied ):
         frefargs=itertools.repeat(None)

      ftest = itertools.repeat(ftest)
      fref  = itertools.repeat(fref)

      #print(ftest,fargs,fref,frefargs,truedata)
      
      for _i,(_ftest,_fargs,_fref,_frefargs,_truedata) in enumerate(zip(ftest,fargs,fref,frefargs,truedata)):
         # if test data is not supplied call _fref to get _truedata
         if ( not data_supplied ):
            _truedata = _fref(*_frefargs)

         #print(_fargs)    
         _actualdata=_ftest(*_fargs)
         # now iterate over the reference data  and actualdata
         for _j,(_td,_ad,_dmsg) in enumerate(zip(_truedata,_actualdata,datamsg)):
            # check if the arrays are are close
            boolclose = npclose(_td,_ad)
            if ( not boolclose ):
               print(f'datamsg=','actual=',_ad,'true=',_td)
               idx,aa,bb = get_mismatch(_td,_ad,closetol=closetol)
               mismsg    = make_mismatch_message(idx,aa,bb)
               outmsg    = optmsg + f"Data not close for field {_dmsg}" + mismsg
               breakpoint()
            self.assertTrue(boolclose,msg=outmsg)


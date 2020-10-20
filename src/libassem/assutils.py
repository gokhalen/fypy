from ..libelem import *
from typing import Union,Tuple
import functools

TOUTGETELM = Union[LinElas1D]

def getelem(elemname:str,ninteg,gdofn)->TOUTGETELM:
    
    if ( elemname == 'linelas2d'):
        return LinElas2D(ninteg=ninteg,gdofn=gdofn)

    if ( elemname == 'linelas2dnumba'):
        return LinElas2DNumba(ninteg=ninteg,gdofn=gdofn)

    if ( elemname == 'linelastrac2d'):
        return LinElasTrac2D(ninteg=ninteg,gdofn=gdofn)

    if ( elemname == 'linelas1d'):
        return LinElas1D(ninteg=ninteg,gdofn=gdofn)

    raise RuntimeError(f'element {elemname} not found')

class KKRhs():
    '''
     this class holds the element stiffness matrices and right hand size
     since the elements return sparse matrices and rhs vectors
     which have dimensions of total degrees of freedom, they're effectively 
     global matrices and rhs vectors which can be added together 
     to get the 'assembled' matrix and rhs
    
     conceptually, we want to drive the assembly step by the following lines of code
 
     kk  = sum(mapelem,iter(fypymesh))
     rhs = sum(mapelem,iter(fypymesh))

     instead of iterating over the elements twice, and localizing data twice
     we can combine both steps into one

     kkrhs = sum(mapelem,iter(fypymesh))

     this requires an object which holds kk and rhs and supports the '+' operator
     that object is this KKRhs
 
    '''
    def __init__(self,gdofn):
        # initialize to zero vector and matrix of dimension gdofn
        self.gdofn = gdofn
        data = (0.0,); row = (0,); col = (0,); tt = (data,(row,col))
        self.kk    = scipy.sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64');
        self.rhs   = scipy.sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64');

    def __add__(self,obj):
        out = KKRhs(self.gdofn)
        out.kk  = self.kk  + obj.kk
        out.rhs = self.rhs + obj.rhs
        return out


class KKRhsRaw():
    '''
    class on similar lines to KKRhs but it works with the raw data, rather than SciPy matrices
    '''
    def __init__(self):
        self.kdata = np.asarray([0])
        self.krow  = np.asarray([0])
        self.kcol  = np.asarray([0])
        
        self.rhsdata = np.asarray([0])
        self.rhsrow  = np.asarray([0])
        self.rhscol  = np.asarray([0])

    def __add__(self,obj):
        out = KKRhsRaw()
        out.kdata = np.concatenate([self.kdata,obj.kdata])
        out.krow  = np.concatenate([self.krow,obj.krow])
        out.kcol  = np.concatenate([self.kcol,obj.kcol])
        
        out.rhsdata = np.concatenate([self.rhsdata,obj.rhsdata])
        out.rhsrow  = np.concatenate([self.rhsrow,obj.rhsrow])
        out.rhscol  = np.concatenate([self.rhscol,obj.rhscol])
        
        return out

def mapelem(tt):
    elem = getelem(tt.eltype,tt.ninteg,tt.gdofn)
    elem.setdata(coord=tt.coord  , prop=tt.prop, bf=tt.bf,      pforce=tt.pforce,
                 dirich=tt.dirich, trac=tt.trac, ideqn=tt.ideqn                 )
    elem.compute()

    kkrhs_e  = KKRhsRaw()
    
    kkrhs_e.kdata = elem.kdata
    kkrhs_e.krow  = elem.krow
    kkrhs_e.kcol  = elem.kcol

    kkrhs_e.rhsdata = elem.fdata
    kkrhs_e.rhsrow  = elem.frow
    kkrhs_e.rhscol  = elem.fcol
    
    # kkrhs_e     = KKRhs(tt.gdofn)
    # kkrhs_e.kk  = elem.kmatrix
    # kkrhs_e.rhs = elem.rhs 
    # return (elem.kmatrix,elem.rhs)
    return kkrhs_e


    


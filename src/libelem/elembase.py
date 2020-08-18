import numpy as np,itertools
from scipy import sparse,linalg
from ..libinteg.gausslegendre import *
from ..libshape.shape import *
from ..libshape.jacobian import *

class ElemBase():

    def __init__(self,ninteg=3,gdofn=None):
        
        # eltype: element type, string
        # ninteg: integration points, integer
        # gdofn : number of global degrees of freedom in the system
        self.ninteg = ninteg
        self.gdofn  = gdofn
        
        # create sparse matrix and rhs
        self.kmatrix = sparse.coo_matrix((gdofn,gdofn),dtype='float64')
        self.rhs     = sparse.coo_matrix((gdofn,1),dtype='float64')

        # move to super
        self.edofn      = self.elnodes*self.elndofn     # total dofn in this element
        self._coord     = np.zeros(self.elnodes*3).reshape(self.elnodes,3)
        self.erhs       = np.zeros(self.edofn)          # rhs vector                                                                
        self.estiff     = np.zeros(self.edofn*self.edofn).reshape(self.edofn,self.edofn)          # element stiffness matrix
        self._prop      = np.zeros(self.elnodes*self.nprop).reshape(self.elnodes,self.nprop)      # stiffness
        self.bf         = np.zeros(self.elnodes*self.ndime).reshape(self.elnodes,self.ndime)      # body force
        self.trac       = np.zeros(self.elnodes*self.elndofn).reshape(self.elnodes,self.elndofn)  # traction
        self.dirich     = np.zeros(self.elnodes*self.elndofn).reshape(self.elnodes,self.elndofn)  # dirichlet data
        

        # data processing arrays, id,ien
        # ideqn(local node number, local dofn) -> global equation number
        # ideqn >= 0 if (node,dofn) is not a dirichlet dofn -1 other wise
        self.ideqn  = np.zeros(self.elnodes*self.elndofn).reshape(self.elnodes,self.elndofn)
        # isbc = 0 if not a bc, 1 if a dirichlet bc and 2 if a traction bc
        self.isbc   = np.zeros(self.elnodes*self.elndofn).reshape(self.elnodes,self.elndofn)

        # get gauss and shape
        self.gg   = getgauss(ndim=self.ndime,npoints=self.ninteg)
        fshape    = getshape(ndim=self.ndime)
        *self.ss, = map(fshape,self.gg.pts)

    @property
    def coord(self):
        return self._coord

    @coord.setter
    def coord(self,x):
        # when coords are set, jacobian is called
        # coordinates have to be of the right shape
        msg = f'In elembase.py coords not of the right shape, expected ({self.elnodes,3}) got {x.shape}'
        assert (x.shape == (self.elnodes,3)), msg
        self._coord = x

    @property
    def prop(self):
        return self._prop

    @prop.setter
    def prop(self,x):
        # properties have to be of the right shape
        msg = f'In elembase.py prop not of the right shape, expected ({self.elnodes},{self.nprop}) got {x.shape}'
        assert (x.shape == (self.elnodes,self.nprop)),msg
        self._prop = x

    def interp_prop(self):
        # interpolate properties at integration points
        self.propinterp = interp_parent(self.prop,self.ss) 
        
    def getjaco(self):
        fjaco = eval(f'jaco{self.ndime}d')
        itr_  = itertools.repeat(self.coord)
        der_  = [s.der for s in self.ss]
        *self.jj, = map(fjaco,itr_,der_)
        

        

        

 

    



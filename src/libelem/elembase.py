import numpy as np,itertools, copy
from scipy import sparse,linalg
from ..libinteg.gausslegendre import *
from ..libinteg.integrate import *
from ..libshape.shape import *
from ..libshape.jacobian import *

# return a reference, accept a copy

class ElemBase():

    # MAKE slots disallow adding new variables

    def __init__(self,ninteg=3,gdofn=None):
        
        # eltype: element type, string
        # ninteg: integration points, integer
        # gdofn : number of global degrees of freedom in the system
        self.ninteg = ninteg
        self.gdofn  = gdofn
        
        # create sparse matrix - declared because these will be put into __slots__ in the future
        data =(0,); row = (0,); col = (0,)
        tt = (data,(row,col))
        self.kmatrix = sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64')
        self.rhs     = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64')
        

        self.edofn   = self.elnodes*self.elndofn                                               # total dofn in this element

        self.erhs       = np.zeros(self.edofn)                                                    # rhs vector
        self.estiff     = np.zeros(self.edofn*self.edofn).reshape(self.edofn,self.edofn)          # element stiffness matrix
        self.erhsbf     = np.zeros(self.edofn)                                                    # rhs vector contribution of body force
        self.erhsdir    = np.zeros(self.edofn)                                                    # rhs vector for dirichlet bc
        self.erhstrac   = np.zeros(self.edofn)                                                    # rhs vector for traction data
        self.erhspf     = np.zeros(self.edofn)

        self._coord     = np.zeros(self.elnodes*3).reshape(self.elnodes,3)                        # coordinates
        self._prop      = np.zeros(self.elnodes*self.nprop).reshape(self.elnodes,self.nprop)      # stiffness property
        self._bf        = np.zeros(self.elnodes*self.elndofn).reshape(self.elnodes,self.elndofn)    # body force property
        self._trac      = np.zeros(self.elnodes*self.elndofn).reshape(self.elnodes,self.elndofn)  # traction propery
        self._dirich    = np.zeros(self.elnodes*self.elndofn).reshape(self.elnodes,self.elndofn)  # dirichlet data
        self._pforce    = np.zeros(self.elnodes*self.elndofn).reshape(self.elnodes,self.elndofn)  # point force data

        # data processing arrays, id,ien
        # ideqn(local node number, local dofn) -> global equation number
        # ideqn >= 0 if (node,dofn) is not a dirichlet dofn -1 other wise
        self._ideqn  = np.zeros(self.elnodes*self.elndofn).reshape(self.elnodes,self.elndofn)
        # isbc = 0 if not a bc, 1 if a dirichlet bc and 2 if a traction bc and 3 if a point force bc
        self._isbc   = np.zeros(self.elnodes*self.elndofn).reshape(self.elnodes,self.elndofn)
        
        # get gauss and shape
        self.gg   = getgauss(ndim=self.ndime,npoints=self.ninteg)
        fshape    = getshape(ndim=self.ndime)
        *self.ss, = map(fshape,self.gg.pts)

    @property
    def isbc(self):
        return self._isbc

    @isbc.setter
    def isbc(self,x):
        msg = f'In elembase.py isbc not of the right shape, expected ({self.elnodes},{self.elndofn}) got {x.shape}'
        assert ( x.shape == (self.elnodes,self.elndofn) ),msg

        boolcmp = np.all( x >=0 ) and np.all(x <=3 )
        assert boolcmp, 'Invalid entries in isbc'

        # if all entries are 1 then boolcmp is true, not boolcmp is false and assertion fails
        boolcmp = np.all( x == 1 )
        assert (not boolcmp),'All entries are constrained'
        
        self._isbc = copy.deepcopy(x)

    @property
    def ideqn(self):
        return self._ideqn

    @ideqn.setter
    def ideqn(self,x):
        # check for right shape
        
        msg = f'In elembase.py ideqn not of the right shape, expected ({self.elnodes},{self.elndofn}) got {x.shape}'
        assert ( x.shape == (self.elnodes,self.elndofn) ),msg

        # check that all entries are less than gdofn  and >=0
        boolcmp = np.all( x < self.gdofn )
        assert boolcmp, 'All entries for ideqn are not less than gdofn'

        # if all entries are less than zero, then boolcmp is true
        boolcmp = np.all( x < 0 )
        assert (not boolcmp),'All entries in ideqn are less than zero'
        
        self._ideqn = copy.deepcopy(x)

    # need to add a setter getter for pforce
    @property
    def pforce(self):
        return self._pforce

    @pforce.setter
    def pforce(self,x):
        msg = f'In elembase.py pforce not of the right shape, expected ({self.elnodes},{self.elndofn}) got {x.shape}'
        self._pforce = copy.deepcopy(x)
    
    @property
    def dirich(self):
        return self._dirich

    @dirich.setter
    def dirich(self,x):
        msg = f'In elembase.py dirich not of the right shape, expected ({self.elnodes},{self.elndofn}) got {x.shape}'
        assert ( x.shape == (self.elnodes,self.elndofn) ),msg
        self._dirich = copy.deepcopy(x)

    @property
    def trac(self):
        return self._trac

    @trac.setter
    def trac(self,x):
        msg = f'In elembase.py trac not of the right shape, expected ({self.elnodes},{self.elndofn}) got {x.shape}'
        assert (x.shape == (self.elnodes,self.elndofn)),msg
        self._trac = copy.deepcopy(x)

    @property
    def bf(self):
        return self._bf

    @bf.setter
    def bf(self,x):
        msg = f'In elembase.py bf not of the right shape, expected ({self.elnodes},{self.elndofn}) got {x.shape}'
        assert (x.shape == (self.elnodes,self.elndofn)),msg
        self._bf = copy.deepcopy(x)
        
    @property
    def coord(self):
        return self._coord

    @coord.setter
    def coord(self,x):
        # when coords are set, jacobian is called
        # coordinates have to be of the right shape
        msg = f'In elembase.py coords not of the right shape, expected ({self.elnodes,3}) got {x.shape}'
        assert (x.shape == (self.elnodes,3)), msg
        self._coord = copy.deepcopy(x)

    @property
    def prop(self):
        return self._prop

    @prop.setter
    def prop(self,x):
        # properties have to be of the right shape
        msg = f'In elembase.py prop not of the right shape, expected ({self.elnodes},{self.nprop}) got {x.shape}'
        assert (x.shape == (self.elnodes,self.nprop)),msg
        self._prop = copy.deepcopy(x)

    def interp(self):
        self.propinterp = interp_parent(self.prop,self.ss)   # interpolate material properties at integration points
        self.bfinterp   = interp_parent(self.bf,self.ss)     # interpolate body force
        self.tracinterp = interp_parent(self.trac,self.ss)   # interp traction
        
    def getjaco(self):
        fjaco = eval(f'jaco{self.ndime}d')
        itr_  = itertools.repeat(self.coord)
        der_  = [s.der for s in self.ss]
        *self.jj, = map(fjaco,itr_,der_)


    def compute_stiffness(self):
        self.estiff = integrate_parent(self.stiffness_kernel,self.gg,self.ss,self.propinterp,self.jj)

    def compute_rhs(self):
        # rhs contribution comes from 1) body force 2) traction 3) dirichlet
        # body force contribution
        self.erhsbf    = integrate_parent(self.rhs_bf_kernel,self.gg,self.ss,self.bfinterp,self.jj)
        
        self.erhsdir   = -self.estiff@((self.dirich).reshape(self.edofn))
        
        # all elements  will implement rhs_trac_kernel; continuum elements will return zero; boundary elements will do the correct integration
        self.erhstrac  = integrate_parent(self.rhs_trac_kernel,self.gg,self.ss,self.tracinterp,self.jj)
        
        # all element implement rhs_point_force method; continuum elements will do the right thing; boundary elements will return zero.
        # for 1d elasticity, the traction boundary condition is implemented as a point force
        self.erhspf    = self.rhs_point_force()

        self.erhs = self.erhsbf + self.erhsdir + self.erhstrac + self.erhspf

    def setdata(self,coord=None,prop=None,bf=None,pforce=None,dirich=None,trac=None,ideqn=None,isbc=None):
        self.coord  = coord;
        self.prop   = prop
        self.bf     = bf
        self.pforce = pforce
        self.dirich = dirich
        self.trac   = trac
        self.ideqn  = ideqn
        self.isbc   = isbc

    def create_global_Kf(self):
        # rhs
        row  = self.ideqn.ravel(order='C')
        data = self.erhs.ravel(order='C')

        # filter erhs to exclude dirichlet data
        frow     = [ row[i]  for i,r in enumerate(row) if r >=0 ]
        fdata    = [ data[i] for i,r in enumerate(row) if r >=0 ]
        fcol     = np.zeros(len(fdata))
        tt       = (fdata,(frow,fcol))
        self.rhs = sparse.coo_matrix(tt,shape=(self.gdofn,1),dtype='float64')

        # filter estiff to exclude dirichlet data
        fdata = []; frow=[]; fcol=[]
        for ii,irow in enumerate(row):
            for jj,icol in enumerate(row):
                if ( irow >=0 and icol >=0 ):
                    fdata.append(self.estiff[ii][jj])
                    frow.append(irow)
                    fcol.append(icol)

        tt = (fdata,(frow,fcol))
        self.kmatrix = sparse.coo_matrix(tt,shape=(self.gdofn,self.gdofn),dtype='float64')

    def compute(self):
        self.getjaco()
        self.interp()
        self.compute_stiffness()
        self.compute_rhs()
        self.create_global_Kf()
        #make stiffness, rhs and global stiffness and rhs
        


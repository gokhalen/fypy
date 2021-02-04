import numpy as np,itertools, copy
from scipy import sparse

from ..libinteg import *
from ..libshape import *

class ElemBase():

    # MAKE slots disallow adding new variables

    def __init__(self,ninteg=3,gdofn=None):
        
        # ninteg: integration points, integer
        # gdofn : number of global degrees of freedom in the system
        self.ninteg = ninteg
        self.gdofn  = gdofn
        
        # create sparse matrix - declared because these will be put into __slots__ in the future
        # data =(0,); row = (0,); col = (0,)
        # tt = (data,(row,col))
        # self.kmatrix = sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64')
        # self.rhs     = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64')
        
        self.edofn   = self.elnodes*self.elndofn                                                  # total dofn in this element

        self.erhs       = np.zeros(self.edofn)                                                    # rhs vector
        self.estiff     = np.zeros(self.edofn*self.edofn).reshape(self.edofn,self.edofn)          # element stiffness matrix
        self.erhsbf     = np.zeros(self.edofn)                                                    # rhs vector contribution of body force
        self.erhsdir    = np.zeros(self.edofn)                                                    # rhs vector for dirichlet bc
        self.erhstrac   = np.zeros(self.edofn)                                                    # rhs vector for traction data
        self.erhspf     = np.zeros(self.edofn)

        self.emass      = np.zeros(self.elnodes*self.elnodes).reshape(self.elnodes,self.elnodes)

        # we don't really need to initialize these values, we can just check in the setter and getter
        # self._coord     = np.zeros(self.elnodes*3).reshape(self.elnodes,3)                        # coordinates
        # self._prop      = np.zeros(self.elnodes*self.nprop).reshape(self.elnodes,self.nprop)      # stiffness property
        # self._bf        = np.zeros(self.elnodes*self.elndofn).reshape(self.elnodes,self.elndofn)    # body force property
        # self._trac      = np.zeros(self.elnodes*self.elndofn).reshape(self.elnodes,self.elndofn)  # traction propery
        # self._dirich    = np.zeros(self.elnodes*self.elndofn).reshape(self.elnodes,self.elndofn)  # dirichlet data
        # self._pforce    = np.zeros(self.elnodes*self.elndofn).reshape(self.elnodes,self.elndofn)  # point force data

        # data processing arrays, id,ien
        # ideqn(local node number, local dofn) -> global equation number
        # ideqn >= 0 if (node,dofn) is not a dirichlet dofn -1 other wise
        # self._ideqn  = np.zeros(self.elnodes*self.elndofn).reshape(self.elnodes,self.elndofn)
        
        # get gauss and shape
        self.gg   = getgauss(ndim=self.ndime,npoints=self.ninteg)
        fshape    = getshape(ndim=self.ndime)
        *self.ss, = map(fshape,self.gg.pts)
        self.ss   = tuple(self.ss)
        
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
        
        self._ideqn = x

    # need to add a setter getter for pforce
    @property
    def pforce(self):
        return self._pforce

    @pforce.setter
    def pforce(self,x):
        msg = f'In elembase.py pforce not of the right shape, expected ({self.elnodes},{self.elndofn}) got {x.shape}'
        assert (x.shape == (self.elnodes,self.elndofn)),msg
        self._pforce = x
    
    @property
    def dirich(self):
        return self._dirich

    @dirich.setter
    def dirich(self,x):
        msg = f'In elembase.py dirich not of the right shape, expected ({self.elnodes},{self.elndofn}) got {x.shape}'
        assert ( x.shape == (self.elnodes,self.elndofn) ),msg
        self._dirich = x

    @property
    def trac(self):
        return self._trac

    @trac.setter
    def trac(self,x):
        msg = f'In elembase.py trac not of the right shape, expected ({self.elnodes},{self.elndofn}) got {x.shape}'
        assert (x.shape == (self.elnodes,self.elndofn)),msg
        self._trac = x

    @property
    def bf(self):
        return self._bf

    @bf.setter
    def bf(self,x):
        msg = f'In elembase.py bf not of the right shape, expected ({self.elnodes},{self.elndofn}) got {x.shape}'
        assert (x.shape == (self.elnodes,self.elndofn)),msg
        self._bf = x

    
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


    @property
    def solution(self):
        return self._solution

    @solution.setter
    def solution(self,x):
        msg = f'In elembase.py solution not of the right shape, expected ({self.elnodes},{self.elndofn}) got {x.shape}'
        assert (x.shape == (self.elnodes,self.elndofn)),msg
        self._solution = x

    @property
    def ideqnmass(self):
        return self._ideqnmass
    
    @ideqnmass.setter
    def ideqnmass(self,x):
        msg = f'In elembase.py solution not of the right shape, expected ({self.elnodes},) got {x.shape}'
        assert (x.shape == (self.elnodes,)),msg
        self._ideqnmass = x

    def interp(self):
        self.propinterp = interp_parent(self.prop,self.ss)   # interpolate material properties at integration points
        self.bfinterp   = interp_parent(self.bf,self.ss)     # interpolate body force
        self.tracinterp = interp_parent(self.trac,self.ss)   # interp traction

        
    def getjaco(self):
        fjaco = eval(f'jaco{self.ndime}d')
        itr_  = itertools.repeat(self.coord)
        der_  = [s.der for s in self.ss]
        *self.jj, = map(fjaco,itr_,der_)
        self.jj   = tuple(self.jj)
        
    def compute_stiffness(self):
        self.estiff = integrate_parent(self.stiffness_kernel,self.gg,self.ss,self.propinterp,self.jj)

    def compute_rhs(self):
        # rhs contribution comes from 1) body force 2) traction 3) dirichlet
        # body force contribution
        self.erhsbf    = integrate_parent(self.rhs_bf_kernel,self.gg,self.ss,self.bfinterp,self.jj)

        self.erhsdir   = -self.estiff@((self.dirich).ravel(order='C'))
        
        # all elements  will implement rhs_trac_kernel; continuum elements will return zero; boundary elements will do the correct integration
        self.erhstrac  = integrate_parent(self.rhs_trac_kernel,self.gg,self.ss,self.tracinterp,self.jj)
        
        # all element implement rhs_point_force method; continuum elements will do the right thing; boundary elements will return zero.
        # for 1d elasticity, the traction boundary condition is implemented as a point force
        self.erhspf    = self.rhs_point_force()

        self.erhs = self.erhsbf + self.erhsdir + self.erhstrac + self.erhspf


    def compute_mass_and_strain_forc(self):
        # compute mass matrix
        rho = [1.0]*len(self.gg.pts)  # fake data at every integration point
        self.getjaco()
        self.emass = integrate_parent(self.mass_kernel,self.gg,self.ss,rho,self.jj)
        self.mdata = self.emass.ravel(order='C')
        row,col = np.indices((self.elnodes,self.elnodes))
        row = row.ravel(order='C')
        col = col.ravel(order='C')
        self.mrow = self.ideqnmass[row]
        self.mcol = self.ideqnmass[col]
        # compute the forcing for strain computation
        exx,eyy,exy=self.make_strains(self.solution,self.jj)
        self.exxrhs=integrate_parent(self.strain_kernel,self.gg,self.ss,exx,self.jj).ravel(order='C')
        self.eyyrhs=integrate_parent(self.strain_kernel,self.gg,self.ss,eyy,self.jj).ravel(order='C')
        self.exyrhs=integrate_parent(self.strain_kernel,self.gg,self.ss,exy,self.jj).ravel(order='C')
        self.strainrow = self.ideqnmass
        self.straincol = [0]*self.elnodes
        


    def setdata(self,coord=None,prop=None,bf=None,pforce=None,dirich=None,
                trac=None,ideqn=None,ideqnmass=None,solution=None):
        if (coord     is not None): self.coord     = coord
        if (prop      is not None): self.prop      = prop
        if (bf        is not None): self.bf        = bf
        if (pforce    is not None): self.pforce    = pforce
        if (dirich    is not None): self.dirich    = dirich
        if (trac      is not None): self.trac      = trac
        if (ideqn     is not None): self.ideqn     = ideqn 
        if (solution  is not None): self.solution  = solution
        if (ideqnmass is not None): self.ideqnmass = ideqnmass


    def create_global_Kf(self):
        # efficient version using numpy
        # filter rhs
        row   = self.ideqn.ravel(order='C')
        data  = self.erhs.ravel(order='C')
        mask  = row > -1
        idrow = row[mask]
        idcol = [0]*len(idrow)
        data  = data[mask]
        # tt    = (data,(idrow,idcol))
        
        self.fdata = data
        self.frow  = idrow
        self.fcol  = idcol
        
        # self.rhs = sparse.coo_matrix(tt,shape=(self.gdofn,1),dtype='float64')

        # filter estiff to exclude dirichlet data
        
        mask2         = np.outer(mask,mask)
        rowidx,colidx = np.indices((self.edofn,self.edofn))
        kk2           = self.estiff[mask2].ravel(order='C')
        rowidx2       = rowidx[mask2].ravel(order='C')
        colidx2       = colidx[mask2].ravel(order='C')
        idrow2        = row[rowidx2]
        idcol2        = row[colidx2]
        
        # tt = (kk2,(idrow2,idcol2))
        # self.kmatrix = sparse.coo_matrix(tt,shape=(self.gdofn,self.gdofn),dtype='float64')
        
        self.kdata = kk2
        self.krow  = idrow2
        self.kcol  = idcol2
        
    def create_global_Kf_old(self):
        # less efficient version in pure Python
        # rhs
        row  = self.ideqn.ravel(order='C')
        data = self.erhs.ravel(order='C')

        # filter erhs to exclude dirichlet data
        frow     = [ row[i]  for i,r in enumerate(row) if r >=0 ]
        fdata    = [ data[i] for i,r in enumerate(row) if r >=0 ]
        fcol     = np.zeros(len(fdata))
        tt       = (fdata,(frow,fcol))
        self.rhs = sparse.coo_matrix(tt,shape=(self.gdofn,1),dtype='float64')

        self.fdata = fdata
        self.frow  = frow
        self.fcol  = fcol
        
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
        
        self.kdata = fdata
        self.krow  = frow
        self.kcol  = fcol

    def compute(self):
        self.getjaco()
        self.interp()
        self.compute_stiffness()
        self.compute_rhs()
        self.create_global_Kf()
        


import numpy as np
import scipy.sparse.linalg

from ..libelem.linelas1d import *
from ..libelem.linelas2d import *
from scipy import sparse

from .test import *


class TestLibElem(TestFyPy):
    
    def test_linelas1d(self):

        # the problem we're solving in (d/dx)(k(x) du/dx) = l
        # on (2.3,5.15) k(2.3) = 2.15 and k(5.15) = 7.25

        # for the global problem, once we get there
        # u(2.3) = 1.7 and traction at (5.15) = 2.1
        # body force = 2x^2

        gdofn  = 10
        ninteg = 3
        elas1d = LinElas1D(ninteg=ninteg,gdofn=gdofn)
        coord = np.zeros(6,dtype='float64').reshape(2,3)
        
        coord[0][0] =  2.3;  coord[0][1]=0.0; coord[0][2] = 0.0;
        coord[1][0] =  5.15; coord[1][1]=0.0; coord[1][2] = 0.0;

        # set material properties and body force
        prop   = np.arange(2,dtype='float64').reshape(2,1)
        bf     = np.arange(2,dtype='float64').reshape(2,1)
        pforce = np.arange(2,dtype='float64').reshape(2,1)
        dirich = np.arange(2,dtype='float64').reshape(2,1)
        trac   = np.arange(2,dtype='float64').reshape(2,1)

        isbc   = np.zeros(2).reshape(2,1)
        ideqn  = np.zeros(2).reshape(2,1)
        
        prop[0][0]   = 2.15;  prop[1][0]   = 7.25
        bf[0][0]     = 11.2;  bf[1][0]     = 13.7
        pforce[0][0] = 23.1;  pforce[1][0] = 17.2
        dirich[0][0] = 2.718; dirich[1][0] = 3.14
        trac[0][0]   = 3.1;   trac[1][0]   = 6.7
        
        isbc[0][0]   = 0;     isbc[1][0]   = 1
        ideqn[0][0]  = 1;     ideqn[1][0]  = 4;
        

        elas1d.setdata(coord=coord,prop=prop,bf=bf,pforce=pforce,dirich=dirich,trac=trac,ideqn=ideqn,isbc=isbc)
        elas1d.compute()

        # check stiffness matrix
        kk = 1.6491228070175434
        expstiff = np.asarray( ( (kk,-kk), (-kk,kk) ))
        msg='In test_linelas1d stiffness matrix comparison for linelas1d fails '
        self.compare_iterables(elas1d.estiff,expstiff,msg=msg,desc='')

        # check rhs body force
        expbf = np.asarray(( 17.1475, 18.335 ))
        msg='In test_linelas1d body force rhs comparison for linelas1d fails '
        self.compare_iterables(elas1d.erhsbf,expbf,msg=msg,desc='')

        # check the point force
        exppf = np.asarray(( 23.1,17.2))
        msg = 'In test_linelas1d point force rhs comparison for linelas1d fails'
        self.compare_iterables(elas1d.erhspf,exppf,msg=msg,desc='')

        # check dirichlet force
        expdir = -expstiff@np.asarray((dirich[0][0],dirich[1][0]))
        msg = 'In test_linelas1d dirichlet force rhs comparison for linelas1d fails'
        self.compare_iterables(elas1d.erhsdir,expdir,msg=msg,desc='')

        # check traction force - continuum elements do not implement traction
        exptrac = np.asarray((0.0,0.0))
        msg = 'In test_linelas1d traction force rhs comparison for linelas1d fails'
        self.compare_iterables(elas1d.erhstrac,exptrac,msg=msg,desc='')
                                      
        # check if filtering stiffness/rhs is done properly
        # same data as before, but ideqn changed

        ideqn[0][0] = -1; ideqn[1][0] = 4

        # check matrix we should get just one entry = [[kk]]
        elas1d.setdata(coord=coord,prop=prop,bf=bf,pforce=pforce,dirich=dirich,trac=trac,ideqn=ideqn,isbc=isbc)
        elas1d.compute()

        data = (kk,); row = (4,); col=(4,)
        tt   = (data,(row,col))
        expstiff = sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64')

        error = scipy.sparse.linalg.norm(expstiff-elas1d.kmatrix) 
        self.assertTrue(error < closeatol,msg='Global stiffness matrices do not match in test_linelas1d')

        # check right hand side
        # the total right hand side is
        tmprhs = expbf + exppf + expdir + exptrac
        data = (tmprhs[1],); row = (4,); col=(0,)
        tt = (data,(row,col))
        exprhs = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64')

        error = scipy.sparse.linalg.norm(exprhs-elas1d.rhs)
        self.assertTrue(error < closeatol,msg='Global rhs vectors do not match in test_linelas1d')


    def test_linelas1d_generated_1(self):
        
        # solve a problem
        # d/dx(du/dx) = 0, u(0)=0 and u(1) = 2
        # exact answer: u(x) = x
        
        start   = 0.0
        end     = 1.0
        nelem   = 10
        hh      = (end-start)/nelem
        kk      = 1
        ninteg  = 3
        gdofn   = (nelem+1)-2
        rightbc = 4

        # prepare exact solution
        xcoord   = np.linspace(start,end,nelem+1)
        
        rows     = list(range(0,nelem+1-2))
        cols     = [0]*(nelem+1-2)
        sol      = rightbc*xcoord[1:-1]
        tt       = (sol,(rows,cols))
                         
        expsol   = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64')
        expsol   = expsol.reshape(gdofn,)

        # create global force and matrix
        data =(0,); row = (0,); col = (0,); tt = (data,(row,col))
        grhs     = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64')
        gkmatrix = sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64')


        # quantities which do not change from element to element are outside the loop
        coord  = np.zeros(6,dtype='float64').reshape(2,3)
        prop   = np.zeros(2,dtype='float64').reshape(2,1)
        prop[0][0] = kk; prop[1][0] = kk
        bf     = np.zeros(2,dtype='float64').reshape(2,1)
        pforce = np.zeros(2,dtype='float64').reshape(2,1)
        # dirich changes from element to element
        trac   = np.zeros(2,dtype='float64').reshape(2,1)

        ideqnarray = list(range(-1,nelem))
        ideqnarray[-1] = -1

        
        elas1d = LinElas1D(ninteg=ninteg,gdofn=gdofn)
        
        for i in range(nelem):
            coord[0][0] = xcoord[i]
            coord[1][0] = xcoord[i+1]

            dirich = np.zeros(2,dtype='float64').reshape(2,1)
            if ( i == (nelem -1)):
                dirich[0][0] = 0
                dirich[1][0] = rightbc

            ideqn = np.zeros(2).reshape(2,1)
            ideqn[0][0] = ideqnarray[i]
            ideqn[1][0] = ideqnarray[i+1]

            isbc = np.zeros(2).reshape(2,1)

            if (i == 0):
                isbc[0][0] = 1

            if (i == (nelem-1)):
                isbc[1][0] = 1
                
            elas1d.setdata(coord=coord,prop=prop,bf=bf,pforce=pforce,dirich=dirich,trac=trac,ideqn=ideqn,isbc=isbc)
            elas1d.compute()

            gkmatrix += elas1d.kmatrix
            grhs     += elas1d.rhs

        x,exitCode = scipy.sparse.linalg.bicg(gkmatrix,grhs.todense(),atol=closeatol)

        # breakpoint()

        error = np.linalg.norm(expsol-x) 
        self.assertTrue(error < closeatol,msg='Solutions do not match in test_linelas1d_generated_1')

    def test_linelas1d_generated_2(self):
        # solve a problem with body force

        # d/dx ( du/dx ) = x  on (0,1) and u(0)=0 and u(1)=0
        # u = (x^3 - x)/6

        # the weak form in hughes' book is obtained by weakening
        # the problem being solved here is u,xx + l = 0  u(0)=q1 and u(1) = q2
        # a(w,u) = (w,l) + boundary terms

        # therefore body force l = -x


        start   = 0.0
        end     = 1.0
        nelem   = 10
        hh      = (end-start)/nelem
        kk      = 1
        ninteg  = 3
        gdofn   = (nelem+1)-2
        rightbc = 0

        # prepare exact solution
        xcoord   = np.linspace(start,end,nelem+1)
        
        rows     = list(range(0,nelem+1-2))
        cols     = [0]*(nelem+1-2)
        sol      = [ ((x**3)-x)/6.0 for x in xcoord[1:-1]]
        tt       = (sol,(rows,cols))
                         
        expsol   = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64')
        expsol   = expsol.reshape(gdofn,)

        # create global force and matrix
        data =(0,); row = (0,); col = (0,); tt = (data,(row,col))
        grhs     = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64')
        gkmatrix = sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64')


        # quantities which do not change from element to element are outside the loop
        coord  = np.zeros(6,dtype='float64').reshape(2,3)
        prop   = np.zeros(2,dtype='float64').reshape(2,1)
        prop[0][0] = kk; prop[1][0] = kk
        pforce = np.zeros(2,dtype='float64').reshape(2,1)
        # dirich changes from element to element
        trac   = np.zeros(2,dtype='float64').reshape(2,1)

        ideqnarray = list(range(-1,nelem))
        ideqnarray[-1] = -1

        
        elas1d = LinElas1D(ninteg=ninteg,gdofn=gdofn)
        
        for i in range(nelem):
            coord[0][0] = xcoord[i]
            coord[1][0] = xcoord[i+1]

            dirich = np.zeros(2,dtype='float64').reshape(2,1)
            if ( i == (nelem -1)):
                dirich[0][0] = 0
                dirich[1][0] = rightbc

            ideqn = np.zeros(2).reshape(2,1)
            ideqn[0][0] = ideqnarray[i]
            ideqn[1][0] = ideqnarray[i+1]

            isbc = np.zeros(2).reshape(2,1)

            bf       = np.zeros(2,dtype='float64').reshape(2,1)
            x0       = coord[0][0]; x1 = coord[1][0];
            bf[0][0] = -x0  ; bf[1][0] = -x1; 
            
            if (i == 0):
                isbc[0][0] = 1

            if (i == (nelem-1)):
                isbc[1][0] = 1
                
            elas1d.setdata(coord=coord,prop=prop,bf=bf,pforce=pforce,dirich=dirich,trac=trac,ideqn=ideqn,isbc=isbc)
            elas1d.compute()

            gkmatrix += elas1d.kmatrix
            grhs     += elas1d.rhs

        x,exitCode = scipy.sparse.linalg.bicg(gkmatrix,grhs.todense(),atol=closeatol)
        # print('solution= ',x,'expected solution=',expsol,'rhs=',grhs.todense())
        error = np.linalg.norm(expsol-x) 
        self.assertTrue(error < closeatol,msg='Solutions do not match in test_linelas1d_generated_1')
        

    def test_linelas1d_generated_3(self):
        # solve a problem with traction (point force)
        # the problem is u,xx + l =0 subject to -u,x(0) = h (leftpforce) and u(1) = 0
        # note that traction is implemented as a point force
        
        start   = 0.0
        end     = 1.0
        nelem   = 10
        hh      = (end-start)/nelem
        kk      = 1
        ninteg  = 3
        gdofn   = (nelem+1)-1  #we have one more unknown compared to previous two cases
        rightbc = 0

        leftpforce = 1

        # prepare exact solution: we have one more unknown compared to previous two cases
        xcoord   = np.linspace(start,end,nelem+1)
        
        rows     = list(range(0,nelem+1-1))
        cols     = [0]*(nelem+1-1)
        sol      = [ leftpforce*(1-x) for x in xcoord[0:-1]]
        
        tt       = (sol,(rows,cols))
                         
        expsol   = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64')
        expsol   = expsol.reshape(gdofn,)

        # create global force and matrix
        data =(0,); row = (0,); col = (0,); tt = (data,(row,col))
        grhs     = sparse.coo_matrix(tt,shape=(gdofn,1),dtype='float64')
        gkmatrix = sparse.coo_matrix(tt,shape=(gdofn,gdofn),dtype='float64')


        # quantities which do not change from element to element are outside the loop
        coord  = np.zeros(6,dtype='float64').reshape(2,3)
        prop   = np.zeros(2,dtype='float64').reshape(2,1)
        prop[0][0] = kk; prop[1][0] = kk
        bf       = np.zeros(2,dtype='float64').reshape(2,1)
        # dirich changes from element to element
        trac   = np.zeros(2,dtype='float64').reshape(2,1)

        ideqnarray = list(range(0,nelem+1))
        ideqnarray[-1] = -1
        
        elas1d = LinElas1D(ninteg=ninteg,gdofn=gdofn)
        
        for i in range(nelem):
            coord[0][0] = xcoord[i]
            coord[1][0] = xcoord[i+1]

            dirich = np.zeros(2,dtype='float64').reshape(2,1)

            pforce = np.zeros(2,dtype='float64').reshape(2,1)
            if ( i == 0):
                pforce[0][0] = leftpforce
            
            if ( i == (nelem -1)):
                dirich[0][0] = 0
                dirich[1][0] = rightbc

            ideqn = np.zeros(2).reshape(2,1)
            ideqn[0][0] = ideqnarray[i]
            ideqn[1][0] = ideqnarray[i+1]

            isbc = np.zeros(2).reshape(2,1)
            
            if (i == (nelem-1)):
                isbc[1][0] = 1
                
            elas1d.setdata(coord=coord,prop=prop,bf=bf,pforce=pforce,dirich=dirich,trac=trac,ideqn=ideqn,isbc=isbc)
            elas1d.compute()

            gkmatrix += elas1d.kmatrix
            grhs     += elas1d.rhs

        x,exitCode = scipy.sparse.linalg.bicg(gkmatrix,grhs.todense(),atol=closeatol)
        # print('solution= ',x,'expected solution=',expsol)
        # breakpoint()
        error = np.linalg.norm(expsol-x) 
        self.assertTrue(error < closeatol,msg='Solutions do not match in test_linelas1d_generated_1')


    def test_linelas2d(self):
        pass

    def test_linelastrac2d(self):
        pass
        


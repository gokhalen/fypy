from .elembase import *
import itertools
import numba as nb

class LinElas2D(ElemBase):
    
    def __init__(self,ninteg,gdofn):
        self.elnodes  = 4
        self.elndofn  = 2
        self.ndime    = 2
        self.dimspace = 2
        self.nprop    = 2

        super().__init__(ninteg=ninteg,gdofn=gdofn)

    def stiffness_kernel(self,gausspts,shape,jaco,prop):
        # second algo in Hughes
        kk = np.zeros(64,dtype='float64').reshape(8,8)

        _lambda = prop[0]
        _mu     = prop[1]

        DD  = np.zeros(9,dtype='float64').reshape(3,3)
        DBb = np.zeros(6,dtype='float64').reshape(3,2)
        BDB = np.zeros(4,dtype='float64').reshape(2,2)
                
        DD[0,0] = _lambda + 2*_mu
        DD[1,1] = _lambda + 2*_mu
        DD[2,2] = _mu
        DD[0,1] = _lambda
        DD[1,0] = _lambda


        itr_b = range(1,self.elnodes+1)

        for bb in itr_b:
            B1 = jaco.gder[bb-1][1-1]
            B2 = jaco.gder[bb-1][2-1]

            DBb[0][0] = DD[0][0]*B1 ; DBb[0][1] = DD[0][1]*B2
            DBb[1][0] = DD[0][1]*B1 ; DBb[1][1] = DD[1][1]*B2
            DBb[2][0] = DD[2][2]*B2 ; DBb[2][1] = DD[2][2]*B1

            itr_a = range(1,bb+1)

            for aa in itr_a:
                B1 = jaco.gder[aa-1][1-1]
                B2 = jaco.gder[aa-1][2-1]
                
                BDB[0][0] = B1*DBb[0][0] + B2*DBb[2][0];  BDB[0][1] = B1*DBb[0][1] + B2*DBb[2][1]
                BDB[1][0] = B2*DBb[1][0] + B1*DBb[2][0];  BDB[1][1] = B2*DBb[1][1] + B1*DBb[2][1]
                
                itr_i = range(1,self.ndime+1)
                itr_j = range(1,self.ndime+1)

                # only the upper triangular part and some lower triangular entries
                # (one entry below the diagonal) are determined
                for ii,jj in itertools.product(itr_i,itr_j):
                    ieqn = self.elndofn*(aa-1) + ii - 1
                    jeqn = self.elndofn*(bb-1) + jj - 1
                    kk[ieqn][jeqn] = BDB[ii-1][jj-1]
                    kk[jeqn][ieqn] = BDB[ii-1][jj-1]

        # kk  = kk + kk.T - np.diag(np.diag(kk))
        # kk2 = self.stiffness_kernel_old(gausspts,shape,jaco,prop)

        # if ( (tol := np.linalg.norm(kk-kk2)) > 1e-11):
        #   print(f'Reporting tolerance error in stiffness_kernel {tol}')
        
        # print('new=',kk)
        # print('old=',kk2)

        # sys.exit()
        # breakpoint()
        
        return kk

    def stiffness_kernel_old(self,gausspts,shape,jaco,prop):
        # inefficient, naive stiffness kernel
        
        kk = np.zeros(64,dtype='float64').reshape(8,8)

        itr_a = range(1,self.elnodes+1)
        itr_b = range(1,self.elnodes+1)

        _lambda = prop[0]
        _mu     = prop[1]

        DD = np.zeros(9,dtype='float64').reshape(3,3)
        DD[0][0] = _lambda + 2*_mu
        DD[1][1] = _lambda + 2*_mu
        DD[2][2] = _mu
        DD[0][1] = _lambda
        DD[1][0] = _lambda

        
        for bb in itr_b:
            Nb1 = jaco.gder[bb-1][1-1]
            Nb2 = jaco.gder[bb-1][2-1]

            BB = np.zeros(6,dtype='float64').reshape(3,2)
            
            BB[0][0] = Nb1
            BB[1][1] = Nb2
            BB[2][0] = Nb2
            BB[2][1] = Nb1

            DB = DD@BB

            for aa in itr_a:
                Na1 = jaco.gder[aa-1][1-1]
                Na2 = jaco.gder[aa-1][2-1]

                BA = np.zeros(6,dtype='float64').reshape(3,2)
                
                BA[0][0] = Na1
                BA[1][1] = Na2
                BA[2][0] = Na2
                BA[2][1] = Na1

                BAT = BA.T

                BDB = BAT@DB

                itr_i = range(1,self.ndime+1)
                itr_j = range(1,self.ndime+1)
                
                for ii,jj in itertools.product(itr_i,itr_j):
                    ieqn = self.elndofn*(aa-1) + ii - 1
                    jeqn = self.elndofn*(bb-1) + jj - 1

                    kk[ieqn][jeqn] = BDB[ii-1][jj-1]
                    
                    # breakpoint()
                    # print(f'{aa=} {ii=} {ieqn=} {bb=} {jj=} {jeqn=}')

        return kk

    def rhs_bf_kernel(self,gausspts,shape,jaco,bf):
        # the body force is \int N_A b_i
        assert (bf.shape == (self.elndofn,)),'wrong shape for body force in rhs_bf_kernel in linelas2d'

        N1,N2,N3,N4 = shape.shape
        b1,b2 = bf

        v1 = N1*b1
        v2 = N1*b2
        v3 = N2*b1
        v4 = N2*b2
        v5 = N3*b1
        v6 = N3*b2
        v7 = N4*b1
        v8 = N4*b2
        
        return np.asarray([v1,v2,v3,v4,v5,v6,v7,v8])

    def rhs_trac_kernel(self,gausspts,shape,jaco,trac):
        # traction is computed by special trac elements
        # this method does not compute traction
        # therefore this method returns a zero vector 
        return np.zeros(self.edofn)

    def rhs_point_force(self):
        return self.pforce.reshape(self.edofn)

        

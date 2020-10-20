import numba as nb
import numpy as np
# specify njit function signature later
# will need changing data structures of arguments

@nb.njit(fastmath=True)
def compute_stiffness_nb(ninteg,gg,ss,jj,prop,kk):
    DD  = np.zeros(9,dtype=nb.float64).reshape(3,3)
    DBb = np.zeros(6,dtype=nb.float64).reshape(3,2)
    BDB = np.zeros(4,dtype=nb.float64).reshape(2,2)

    for iinte in range(ninteg*ninteg):
        # get wtjac
        wtjac = gg.wts[iinte]*jj[iinte].jdet
        # interpolate properties

        propinterp = np.zeros(2,dtype=nb.float64)
        for iprop in range(2):
            for inode in range(4):
                propinterp[iprop] += prop[inode][iprop]*ss[iinte].shape[inode]

        # interpolated properties
        _lambda = propinterp[0]
        _mu     = propinterp[1]

        DD[0,0] = _lambda + 2*_mu ; DD[0,0] = DD[0,0]*wtjac 
        DD[1,1] = _lambda + 2*_mu ; DD[1,1] = DD[1,1]*wtjac 
        DD[2,2] = _mu             ; DD[2,2] = DD[2,2]*wtjac 
        DD[0,1] = _lambda         ; DD[0,1] = DD[0,1]*wtjac 
        DD[1,0] = _lambda         ; DD[1,0] = DD[1,0]*wtjac 


        for bb in range(1,5):
            B1 = jj[iinte].gder[bb-1][1-1]
            B2 = jj[iinte].gder[bb-1][2-1]

            DBb[0][0] = DD[0][0]*B1 ; DBb[0][1] = DD[0][1]*B2
            DBb[1][0] = DD[0][1]*B1 ; DBb[1][1] = DD[1][1]*B2
            DBb[2][0] = DD[2][2]*B2 ; DBb[2][1] = DD[2][2]*B1


            for aa in range(1,bb+1):
                B1 = jj[iinte].gder[aa-1][1-1]
                B2 = jj[iinte].gder[aa-1][2-1]

                BDB[0][0] = B1*DBb[0][0] + B2*DBb[2][0];  BDB[0][1] = B1*DBb[0][1] + B2*DBb[2][1]
                BDB[1][0] = B2*DBb[1][0] + B1*DBb[2][0];  BDB[1][1] = B2*DBb[1][1] + B1*DBb[2][1]


                for iii in range(1,3):
                    for jjj in range(1,3):
                        ieqn = 2*(aa-1) + iii - 1
                        jeqn = 2*(bb-1) + jjj - 1
                        if ( jeqn >= ieqn ):
                            kk[ieqn][jeqn] += BDB[iii-1][jjj-1]
                            if ( jeqn != ieqn ):
                                kk[jeqn][ieqn] += BDB[iii-1][jjj-1]




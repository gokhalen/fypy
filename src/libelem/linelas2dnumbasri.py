from .elembase             import *
from .linelas2d            import *
from .linelas2dnumbasrilib import *

import numba as nb
import numpy as np

class LinElas2DNumbaSRI(LinElas2D):
    
    def __init__(self,ninteg,gdofn):
        self.elnodes  = 4
        self.elndofn  = 2
        self.ndime    = 2
        self.dimspace = 2
        self.nprop    = 2
        assert (ninteg == 3),'Only 3 point integration supported for LinElas2DNumbaSRI (hacks for reduced integration depend on it)'
        super().__init__(ninteg=ninteg,gdofn=gdofn)
        
    # override elembases' compute_stiffness for fast numba implementation
    def compute_stiffness(self):
        kk   = np.zeros(8*8,dtype='float64').reshape(8,8)
        # need to break the namedtuples into arrays implicitly ordered by integration point
        tmp  = [s.shape for s in self.ss] ; shp  = np.asarray(tmp)
        tmp  = [ j.gder for j in self.jj] ; gder = np.asarray(tmp)
        tmp  = [ j.jdet for j in self.jj] ; jdet = np.asarray(tmp)
        compute_stiffness_nb_sri(self.ninteg,self.gg.wts,shp,gder,jdet,self.prop,kk)
        self.estiff = kk

    def compute_mass_and_strain_forc(self):
        # compute mass matrix
        rho = [1.0]*len(self.gg.pts)  # fake data at every integration point
        self.getjaco()
        # self.emass = integrate_parent(self.mass_kernel,self.gg,self.ss,rho,self.jj)
        # self.mdata = self.emass.ravel(order='C')

        # numba implementation of mass and strain forcing
        mm_nb     = np.zeros((self.elnodes,self.elnodes),dtype='float64')
        exxrhs_nb = np.zeros((self.elnodes,),dtype='float64')
        eyyrhs_nb = np.zeros((self.elnodes,),dtype='float64')
        exyrhs_nb = np.zeros((self.elnodes,),dtype='float64')
        
        tmp   = [ s.shape for s in self.ss] ; shp  = np.asarray(tmp)
        tmp   = [ j.gder  for j in self.jj] ; gder = np.asarray(tmp)
        tmp   = [ j.jdet  for j in self.jj] ; jdet = np.asarray(tmp)
        
        compute_mass_nb(self.ninteg,self.gg.wts,shp,gder,jdet,self.solution,
                        mm_nb,exxrhs_nb,eyyrhs_nb,exyrhs_nb)
        
        self.mdata = mm_nb.ravel(order='C')
        row,col = np.indices((self.elnodes,self.elnodes))
        row = row.ravel(order='C')
        col = col.ravel(order='C')
        self.mrow = self.ideqnmass[row]
        self.mcol = self.ideqnmass[col]


        # compute the forcing for strain computation
        #exx,eyy,exy=self.make_strains(self.solution,self.jj)
        #self.exxrhs=integrate_parent(self.strain_kernel,self.gg,self.ss,exx,self.jj).ravel(order='C')
        #self.eyyrhs=integrate_parent(self.strain_kernel,self.gg,self.ss,eyy,self.jj).ravel(order='C')
        #self.exyrhs=integrate_parent(self.strain_kernel,self.gg,self.ss,exy,self.jj).ravel(order='C')
        
        self.exxrhs = exxrhs_nb.ravel(order='C')
        self.eyyrhs = eyyrhs_nb.ravel(order='C')
        self.exyrhs = exyrhs_nb.ravel(order='C')
        
        self.strainrow = self.ideqnmass
        self.straincol = [0]*self.elnodes

        #print(exxrhs_nb)
        #print(eyyrhs_nb)
        #print(exyrhs_nb)
        #print(self.exxrhs)
        #print(self.eyyrhs)
        #print(self.exyrhs)

        
        # breakpoint()
        
    

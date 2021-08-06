# A quick and dirty 1d,2d-mesh generator
import os,sys,json;
import numpy as np;
import matplotlib as mpl
 # https://stackoverflow.com/questions/45993879/matplot-lib-fatal-io-error-25-inappropriate-ioctl-for-device-on-x-server-loc See nanounanue's answer
mpl.use('Agg')
import matplotlib.pyplot as plt
import random
import pyvista as pv

class FyPyMesh():
    stflist  = ['homogeneous','inclusion','random']
    elemlist = ['linelas2d','linelas2dnumba','linelas2dnumbasri']
    
    def __init__(self,inputdir='',outputdir=''):
        if ( inputdir != '') and ( inputdir[-1] != '/'):
            print('input directory requires a trailing /')
            sys.exit()

        if ( outputdir != '') and ( outputdir[-1] != '/'):
            print('output directory requires a trailing /')
            sys.exit()
            
        self.inputdir  = inputdir
        self.outputdir = outputdir
    
    def make_eqn_no(self,ideqn):
        # must be called after all negative numbers are set
        ieqnno=0
        for inode in range(0,self.nnodes):
            for idofn in range(0,self.ndofn):
                if (ideqn[inode][idofn] >= 0):
                    ideqn[inode][idofn] = ieqnno
                    ieqnno +=1

        self.gdofn = ieqnno

    def make_ideqnmass(self,ideqnmass):
        # we need ideqn for computing strains
        ieqnno = 0
        for inode in range(0,self.nnodes):
            ideqnmass[inode] = ieqnno
            ieqnno +=1

    def create_mesh_1d(self,start=0.0,end=1.0,nelem=10,stf='homogeneous'):
        self.start    = start
        self.end      = end
        self.length   = self.end - self.start
        self.nelem    = nelem
        self.nelemx   = nelem
        self.nelemy   = 0
        self.nnodex   = nelem+1
        self.stf      = stf
        self.stfmin   = 1
        self.stfmax   = 5

        # some constant data not exposed
        self.ninteg   = 3
        self.nprop    = 1
        self.ndofn    = 1
        self.ndime    = 1

        # derived data
        self.nnodes   = self.nelem  + 1
        assert (end > start),f'{end=} should be greater than {start=}'
        
        if ( not stf in FyPyMesh.stflist ):
            print(f'Unknown value ({stf=}) for stf defaulting to homogeneous')
            self.stf = 'homogeneous'


        coordx = np.linspace(self.start,self.end,self.nelem+1)
        coordy = np.zeros(coordx.shape)
        coordz = np.zeros(coordx.shape)

        *self.coord, = zip(coordx,coordy,coordz)
        self.conn    = [ [ii,ii+1] for ii in range(1,self.nelem+1)]
        
        # add element type to conn
        for cc in self.conn:
            cc.append('linelas1d')

        
        self.prop      = [ [self.stfmin] for i in self.coord ]
        self.ideqn     = [ [0]*self.ndofn for i in self.coord]


        # set the first and last ideqn to negative for dirichlet
        self.ideqn[0][0]  = -1
        self.ideqn[-1][0] = -1

        self.make_eqn_no(self.ideqn)
 
        # set dirichlet bcs
        self.dirich        = [ [0]*self.ndofn for i in self.coord ]
        self.dirich[0][0]  = 1.0
        self.dirich[-1][0] = 2.0
        self.trac          = [ [0]*self.ndofn for i in self.coord ]
        self.pforce            = [ [0]*self.ndofn for i in self.coord ]
        

        # set body force
        self.bf = [ [0]*self.ndofn for i in self.coord ]

        if self.stf == 'inclusion':
            self.prop = [ [self.stfmax] if ( (x >=0.4*self.length) and (x <= 0.6*self.length)) else [self.stfmin] for x in coordx ]

    def create_mesh_2d(self,length=10.0,breadth=10.0,nelemx=10,nelemy=10,
                       stftype = 'homogeneous',bctype='dirich',
                       radii   = [1.0],
                       centers = [[5.0,5.0]],
                       mu      = [2.5],
                       muback  = 1.0,
                       nu      = 0.25,
                       eltype  = 'linelas2dnumba',
                       bcmag   = -0.06
                       ):
        
        assert (length > 0),'length has to be greater than 0'
        assert (breadth > 0),'breadth has to be greater than 0'
        assert (len(radii)==len(centers)),'number of radii and centers are not equal'
        assert (eltype in FyPyMesh.elemlist),'Unknown element specified'
        
        xstart  = 0.0; xend = xstart + length ; dx = length  / nelemx 
        ystart  = 0.0; yend = ystart + breadth; dy = breadth / nelemy

        nnodex  = nelemx + 1;
        nnodey  = nelemy + 1
        
        # data needed for write_mesh
        self.nelem    = nelemx*nelemy
        self.nelemx   = nelemx
        self.nelemy   = nelemy
        self.nnodex   = nelemx+1
        self.nnodey   = nelemy+1
        self.length   = length
        self.breadth  = breadth
        if ( bctype == 'trac'): self.nelem += nelemx  # if traction elements, increase nlelem
        self.nnodes   = nnodex*nnodey
        self.ninteg   = 3
        self.ndofn    = 2
        self.nprop    = 2
        self.ndime    = 2
        self.mu       = mu
        self.muback   = muback
        self.rnu      = 2*nu/(1-2*nu)
        self.centers  = centers     # list of centers for inclusions
        self.radii    = radii  # list of radii   for inclusions

        # values to use for bcs
        qdirich = bcmag
        qtrac   = bcmag
        
        # node numbers increase by 1 in the y direction by nodey in the x-direction
        # node numbers are defined implicitly.

        # create coords
        self.coord=[]
        xx = xstart; yy =ystart; zz=0.0
        for ix in range(1,nnodex+1):
            for iy in range(1,nnodey+1):
                inode = nnodey*(ix-1) + iy
                self.coord.append((xx,yy,zz))
                yy += dy
            xx += dx
            yy  = ystart

        # create connectivity
        self.conn   = []
        for ix in range(1,nelemx+1):
            for iy in range(1,nelemy+1):
                # global element number
                ielem = (ix-1)*nelemy + iy
                # need to get lower left node (n1)
                n1  =  (ix-1)*nnodey + iy
                n2  = n1 + nnodey
                n3  = n2 + 1
                n4  = n1 + 1
                self.conn.append([n1,n2,n3,n4,eltype])

        # add traction elements if necessary
        if ( bctype == 'trac' ):
            n1 = nnodey
            n2 = n1 + nnodey
            for ix in range(nelemx*nelemy+1,self.nelem+1):
                # notice n1 and n2 is reversed; the order in which they are
                # traversed should be the same as in the quad element order
                self.conn.append([n2,n1,'linelastrac2d'])
                n1 += nnodey
                n2 += nnodey

        # create properties
        self.prop   = []; xmid = length/2.0; ymid = breadth/2.0; 
        for x,y,z in self.coord:

            if ( stftype == 'homogeneous'):
                mu = self.muback ; lam = self.rnu*mu
                self.prop.append([lam,mu])
           
            if ( stftype == 'inclusion'):
                mu   = self.muback; lam =self.rnu*self.muback
                for [xcen,ycen],rad,muinc in zip(self.centers,self.radii,self.mu):
                    dist = (x-xcen)**2 + (y-ycen)**2
                    dist = dist**0.5
                    if ( dist <= rad ):
                        mu = muinc; lam = self.rnu*muinc;
                self.prop.append([lam,mu])

            # creates a cross at the center
            if ( stftype == 'cross'):
                mu = self.muback; lam = self.rnu*self.muback
                _cross_height = self.breadth*0.7
                _cross_length = self.length*0.7
                _cross_thick  = 0.2*min(self.length,self.breadth)
                _cross_cenx   = self.length/2.0
                _cross_ceny   = self.breadth/2.0

                # limits of horizontal rectangle
                xmin = _cross_cenx - (_cross_length/2.0)
                xmax = _cross_cenx + (_cross_length/2.0)
                ymin = _cross_ceny - (_cross_thick/2.0)
                ymax = _cross_ceny + (_cross_thick/2.0)

                if (x <= xmax) and (x >= xmin) and (y <= ymax) and (y >= ymin):
                    mu = 4.0; lam = self.rnu*mu

                # liits of horizontal rectangle
                xmin = _cross_cenx - (_cross_thick/2.0)
                xmax = _cross_cenx + (_cross_thick/2.0)
                ymin = _cross_ceny - (_cross_height/2.0)
                ymax = _cross_ceny + (_cross_height/2.0)

                if (x <= xmax) and (x >= xmin) and (y <= ymax) and (y >= ymin):
                    mu = 4.0; lam = self.rnu*mu
                
                self.prop.append([lam,mu])
                
                pass

        # create ideqn and dirichlet bc
        self.ideqn      = [ [0]*self.ndofn for i in self.coord]
        self.ideqnmass  = [ 0              for i in self.coord]
        self.dirich     = [ [0]*self.ndofn for i in self.coord]

        # create lower boundary dirichlet condition
        # first the x-condition on the first node
        self.ideqn[0][0]  = -1;
        self.dirich[0][0] = 0.0;
        # then the y conditions on the lower boundary
        for i in range(1,(self.nnodes-nnodey+1)+1,nnodey):
            self.ideqn[i-1][1]  = -1
            self.dirich[i-1][1] = 0.0

        if (bctype == 'dirich'):
            for i in range(nnodey,self.nnodes+1,nnodey):
                self.ideqn[i-1][1]  = -1
                self.dirich[i-1][1] = qdirich
                # make the rest of the equation numbers

        # must be called after all negative numbers are set
        self.make_eqn_no(self.ideqn)
        self.make_ideqnmass(self.ideqnmass)
        
        self.bf   = [ [0]*self.ndofn for i in self.coord ]
        self.pforce   = [ [0]*self.ndofn for i in self.coord ]
        self.trac = [ [0]*self.ndofn for i in self.coord ]

        if (bctype == 'trac'):
            for i in range(nnodey,self.nnodes+1,nnodey):
                self.trac[i-1][1] = qtrac

        # dump out mu and lambda 
    

    # not supported any more - json dumps are more elegant
    def write_field(self,field_name,data,fout):
        fout.write('$'+field_name+'\n')
        for idx,dd in enumerate(data):
            fout.write(str(idx+1)+' ')
            for d in dd:
                fout.write(str(d) +' ')
            fout.write('\n')
        fout.write('$'+field_name+'\n')

    # not supported anymore - json dumps are more elegant
    def write_mesh(self,filename='data.in'):
        # this code should be independent of dimension
        with open(filename,'w') as fout:
            fout.write('# fypy mesh generator\n')


            # some global data
            fout.write('nelem='+str(self.nelem)+' nelemx='+str(self.nelemx)+ ' nelemy='+str(self.nelemy)+
                       ' nnodes='+str(self.nnodes)+' ninteg='+str(self.ninteg)+
                       ' ndofn='+str(self.ndofn)+ ' nprop='+str(self.nprop)+' ndime='+str(self.ndime)+
                       ' gdofn='+str(self.gdofn)+'\n')
            
            # write fields
            self.write_field('coord',self.coord,fout)
            self.write_field('conn',self.conn,fout)
            self.write_field('prop',self.prop,fout)
            self.write_field('ideqn',self.ideqn,fout)
            self.write_field('dirich',self.dirich,fout)
            self.write_field('bf',self.bf,fout)
            self.write_field('trac',self.trac,fout)
            self.write_field('pf',self.pforce,fout)

    def json_dump(self,filename='data.in.json'):
        dd = { 'nelem':self.nelem,
               'nelemx':self.nelemx,
               'nelemy':self.nelemy,
               'nnodex':self.nnodex,
               'nnodey':self.nnodey,
               'nnodes':self.nnodes,
               'ninteg':self.ninteg,
               'ndofn':self.ndofn,
               'nprop':self.nprop,
               'ndime':self.ndime,
               'gdofn':self.gdofn,
               'coord':self.coord,
               'conn':self.conn,
               'prop':self.prop,
               'ideqn':self.ideqn,
               'ideqnmass':self.ideqnmass,
               'dirich':self.dirich,
               'bf':self.bf,
               'trac':self.trac,
               'pforce':self.pforce
             }

        with open(self.outputdir+filename,'x') as fout:
            json.dump(dd,fout,indent=4)

    # used to read the mesh into the FEM code
    def json_read(self,filename='data.json'):
        with open(self.inputdir+filename,'r') as fin:
            jj=json.load(fin)

        # recreate mesh data structure
        for key,value in jj.items():
            setattr(self,key,value)

    def make_solution_from_rhs(self,rhs):
        self.solution = np.zeros(self.nnodes*self.ndofn).reshape(self.nnodes,self.ndofn)

        for inode in range(1,self.nnodes+1):
            for idofn in range(1,self.ndofn+1):
                ieqnno = self.ideqn[inode-1][idofn-1]
                if ( ieqnno >= 0):
                    self.solution[inode-1][idofn-1] = rhs[ieqnno]
                else:
                    self.solution[inode-1][idofn-1] = self.dirich[inode-1][idofn-1]

    def make_output(self,outfile):
        # output the solution only
        # rest can be picked up from input file
        
        sollist = [ list(ss) for ss in self.solution ]
        exxlist = list(self.exx)
        eyylist = list(self.eyy) 
        exylist = list(self.exy)         

        dd = {'solution':sollist,
              'exx':exxlist,
              'eyy':eyylist,
              'exy':exylist
             }

        with open(self.outputdir+outfile,'w') as fout:
            json.dump(dd,fout,indent=4)

    def make_sparsity(self):
        # row, col
        pass

    def preprocess(self,suffix):
        lam    = np.asarray(self.prop)[:,0]
        mu     = np.asarray(self.prop)[:,1]
        
        stfminmu  = np.min(mu) ; stfmaxmu  = np.max(mu)
        stfminlam = np.min(lam) ; stfmaxlam = np.max(lam)
        
        self.plotfield(self.coord, lam, 'lambda', stfminlam, stfmaxlam, suffix=suffix+'_fypy')
        self.plotfield(self.coord, mu,  'mu' , stfminmu, stfmaxmu, suffix=suffix+'_fypy')

    def postprocess(self,suffix):
        
        ux  = np.asarray(self.solution)[:,0]
        uy  = np.asarray(self.solution)[:,1]
        exx = np.asarray(self.exx)
        eyy = np.asarray(self.eyy)
        exy = np.asarray(self.exy)
  
        self.plotfield(self.coord, ux, 'ux', suffix=suffix+'_fypy')
        self.plotfield(self.coord, uy, 'uy', suffix=suffix+'_fypy')

        # adjust the scale on the strain plots so that the
        # spurious strain on the fixed node is not seen
        # the maximum scale has to be pushed towards +infty
        # the minimum scale has to be pused towards  -infty
        ftol   = 0.00001
        exxmin = np.min(exx)-ftol; exxmax = np.max(exx)+ftol;
        eyymin = np.min(eyy)-ftol; eyymax = np.max(eyy)+ftol;
        exymin = np.min(exy)-ftol; exymax = np.max(exy)+ftol;

        
        self.plotfield(self.coord, exx,'exx',fmin=exxmin,fmax=exxmax,suffix=suffix+'_fypy')
        self.plotfield(self.coord, eyy,'eyy',fmin=eyymin,fmax=eyymax,suffix=suffix+'_fypy')
        self.plotfield(self.coord, exy,'exy',fmin=exymin,fmax=exymax,suffix=suffix+'_fypy')
            
    def plotfield(self,coord,field,fieldname,fmin=None,fmax=None,suffix=''):
        # do not try .reshape(self.nnodey,self.nnodex)
        # if you want to switch dimensions, then take the transpose
        xx    = np.asarray(coord)[:,0].reshape(self.nnodex,self.nnodey)
        yy    = np.asarray(coord)[:,1].reshape(self.nnodex,self.nnodey)
        field = field.reshape(self.nnodex,self.nnodey) 

        plt.figure(fieldname)
        plt.pcolormesh(xx,yy,field)
        plt.clim([fmin,fmax])
        plt.colorbar()
        plt.title(fieldname)
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(self.outputdir+fieldname+f'{suffix}.png')
        # without this close figure instances 'stay alive'
        # and multiple colorbars are drawn
        # putting plt.close() in init doesn't seem to work
        plt.close()

    def postprocess_pv(self,suffix):
        # create vertices
        vertices = np.asarray(self.coord)
        # create quad faces - check if the element is a quad len(ff)==5
        # drop the last element because it is a string describing element type
        faces = [ [4,*ff[:-1]] for ff in self.conn if len(ff)==5 ]
        faces = np.asarray(faces)
        faces[:,1:] -=1             # subtract 1 because connectivity is 1 based
        surf = pv.PolyData(vertices,faces)
        
        # create data to plot
        ux     = np.asarray(self.solution)[:,0]
        uy     = np.asarray(self.solution)[:,1]
        lam    = np.asarray(self.prop)[:,0]
        mu     = np.asarray(self.prop)[:,1]
        stfmin = np.min([lam,mu]) ; stfmax = np.max([lam,mu])
        
        surf.point_arrays['ux']  = ux
        surf.point_arrays['uy']  = uy
        surf.point_arrays['lam'] = lam
        surf.point_arrays['mu']  = mu

        self.plot_pyvista(surf,ux,'ux',clim=None,suffix=suffix);
        self.plot_pyvista(surf,uy,'uy',clim=None,suffix=suffix);
        self.plot_pyvista(surf,lam,'lam',clim=None,suffix=suffix);
        self.plot_pyvista(surf,mu,'mu',clim=None,suffix=suffix);


    def plot_pyvista(self,surf,field,fieldname,clim=None,suffix=''):
        # you can pass min and max range in clim, as in clim=[min,max]
        surf.point_arrays[fieldname]=field
        sargs = dict(height=0.25, vertical=True, position_x=0.85, position_y=0.05)
                
        p = pv.Plotter()
        p.show_axes()
        p.add_mesh(surf,clim=clim,scalars=fieldname,scalar_bar_args=sargs)
        p.view_xy()
        p.show(screenshot=self.outputdir+fieldname+'_pyvista.png')
        p.close()

        
    



 
            



        
    

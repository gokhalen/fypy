# A quick and dirty 1d,2d-mesh generator
import sys,json;
import numpy as np;

class FyPyMesh():
    stflist = ['homogeneous','inclusion']
    
    def __init__(self):
        pass
    
    def make_eqn_no(self,ideqn):
        # must be called after all negative numbers are set
        ieqnno=0
        for inode in range(0,self.nnodes):
            for idofn in range(0,self.ndofn):
                if (ideqn[inode][idofn] >= 0):
                    ideqn[inode][idofn] = ieqnno
                    ieqnno +=1

        self.gdofn = ieqnno
                    

    def create_mesh_1d(self,start=0.0,end=1.0,nelem=10,stf='homogeneous'):
        self.start    = start
        self.end      = end
        self.length   = self.end - self.start
        self.nelem    = nelem
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

        
        self.prop    = [ [self.stfmin] for i in self.coord ]

        self.ideqn   = [ [0]*self.ndofn for i in self.coord]

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

    def create_mesh_2d(self,length=10.0,breadth=10.0,nelemx=10,nelemy=10,stf='homogeneous',bctype='dirich'):
        assert (length > 0),'length has to be greater than 0'
        assert (breadth > 0),'breadth has to be greater than 0'
        
        xstart  = 0.0; xend = xstart + length ; dx = length  / nelemx 
        ystart  = 0.0; yend = ystart + breadth; dy = breadth / nelemy

        nnodex  = nelemx + 1;
        nnodey = nelemy + 1
        
        # data needed for write_mesh
        self.nelem    = nelemx*nelemy
        if ( bctype == 'trac'): self.nelem += nelemx  # if traction elements, increase nlelem
        self.nnodes   = nnodex*nnodey
        self.ninteg   = 3
        self.ndofn    = 2
        self.nprop    = 2
        self.ndime    = 1
        
        # parameters 
        self.stfmin   = 1
        self.stfmax   = 5

        qdirich = -1
        qtrac   = -1
        
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
                self.conn.append([n1,n2,n3,n4,'linelas2d'])

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
        self.prop   = []; xmid = length/2.0; ymid = breadth/2.0; rad = length*0.2
        for x,y,z in self.coord:
            dist = (x-xmid)**2 + (y-ymid)**2
            dist = dist**0.5
            mu = self.stfmin; lam = 2.0*self.stfmin;
            
            if ( stf=='inclusion' and ( dist < rad ) ):
                mu = self.stfmax; lam =2.0*self.stfmax

            self.prop.append([lam,mu])

        # create ideqn and dirichlet bc
        self.ideqn  = [ [0]*self.ndofn for i in self.coord]
        self.dirich = [ [0]*self.ndofn for i in self.coord]

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


        
        self.bf   = [ [0]*self.ndofn for i in self.coord ]
        self.pforce   = [ [0]*self.ndofn for i in self.coord ]
        self.trac = [ [0]*self.ndofn for i in self.coord ]

        if (bctype == 'trac'):
            for i in range(nnodey,self.nnodes+1,nnodey):
                self.trac[i-1][1] = qtrac

    def write_field(self,field_name,data,fout):
        fout.write('$'+field_name+'\n')
        for idx,dd in enumerate(data):
            fout.write(str(idx+1)+' ')
            for d in dd:
                fout.write(str(d) +' ')
            fout.write('\n')
        fout.write('$'+field_name+'\n')

    def write_mesh(self,filename='data.in'):
        # this code should be independent of dimension
        with open(filename,'w') as fout:
            fout.write('# fypy mesh generator\n')


            # some global data
            fout.write('nelem='+str(self.nelem)+' nnodes='+str(self.nnodes)+' ninteg='+str(self.ninteg)+
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
               'dirich':self.dirich,
               'bf':self.bf,
               'trac':self.trac,
               'pforce':self.pforce
             }
        with open(filename,'w') as fout:
            json.dump(dd,fout,indent=4)

    def json_read(self,filename='data.json'):
        with open(filename,'r') as fin:
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

 
            



        
    

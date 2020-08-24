# A quick and dirty 1d,2d-mesh generator
import sys;
import numpy as np;

class FyPyMesh():
    stflist = ['homogeneous','inclusion']
    
    def __init__(self,):
        pass

    def create_mesh_1d(self,start=0.0,end=1.0,nelem=10,stf='homogeneous',filename='data.in'):
        self.start    = start
        self.end      = end
        self.length   = self.end - self.start
        self.nelem    = nelem
        self.stf      = stf
        self.filename = filename
        self.stfmin   = 1
        self.stfmax   = 5

        # some constant data not exposed
        self.ninteg   = 3
        self.nprop    = 1
        self.ndofn    = 1

        # derived data
        self.nnodes   = self.nelem  + 1
        self.gdofn    = self.nnodes - 2   # we're typically solving a dirichlet conditions at both end points

        assert (end > start),f'{end=} should be greater than {start=}'
        
        if ( not stf in FyPyMesh.stflist ):
            print(f'Unknown value ({stf=}) for stf defaulting to homogeneous')
            self.stf = 'homogeneous'

        self.coordx = np.linspace(self.start,self.end,self.nelem)
        self.coordy = np.zeros(self.coordx.shape)
        self.coordz = np.zeros(self.coordx.shape)

        *self.coord, = zip(self.coordx,self.coordy,self.coordz) 
        self.conn    = [ [ii,ii+1] for ii in range(1,self.nelem+1)]
        self.prop    = [ [self.stfmin] for i in self.coordx ]

        # set dirichlet bcs
        self.dirich        = [ [0]*self.ndofn for i in self.coordx ]
        self.dirich[0][0]  = 1.0
        self.dirich[-1][0] = 2.0
        self.trac          = [ [0]*self.ndofn for i in self.coordx ]
        self.pf            = [ [0]*self.ndofn for i in self.coordx ]
        

        # set body force
        self.bf = [ [0]*self.ndofn for i in self.coordx ]

        if self.stf == 'inclusion':
            self.prop = [ [self.stfmax] if ( (x >=0.4*self.length) and (x <= 0.6*self.length)) else [self.stfmin] for x in self.coordx ]

    def create_mesh_2d(self,length=10.0,breadth=10.0,nelemx=10,nelemy=10,stf='homogeneous',filename='data.in'):
        assert (length > 0),'length has to be greater than 0'
        assert (breadth > 0),'breadth has to be greater than 0'
        
        xstart  = 0.0; xend = xstart + length
        ystart  = 0.0; yend = ystart + breadth


    def write_field(self,field_name,data,fout):
        fout.write('$'+field_name+'\n')
        for idx,dd in enumerate(data):
            fout.write(str(idx+1)+' ')
            for d in dd:
                fout.write(str(d) +' ')
            fout.write('\n')
        fout.write('$'+field_name+'\n')

    def write_mesh(self):
        
        # this code should be independent of dimension
        with open(self.filename,'w') as fout:
            fout.write('# fypy mesh generator\n')

            # need to add gdofn
            # some global data
            fout.write('nelem='+str(self.nelem)+' nnodes='+str(self.nnodes)+' ninteg='+str(self.ninteg)+
                       ' ndofn='+str(self.ndofn)+ ' nprop='+str(self.nprop)+' gdofn='+str(self.gdofn)+
                       '\n')
            
            # write fields
            self.write_field('coord',self.coord,fout)
            self.write_field('conn',self.conn,fout)
            self.write_field('prop',self.prop,fout)
            self.write_field('dirich',self.dirich,fout)
            self.write_field('bf',self.bf,fout)
            self.write_field('trac',self.trac,fout)
            self.write_field('pf',self.pf,fout)
            # write point force

    def read_mesh(self,filename):
        # should be independent of dimension
        pass





        
    

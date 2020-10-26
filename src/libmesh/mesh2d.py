from fypymesh import *

if __name__ =='__main__':
    mesh2d = FyPyMesh()
    mesh2d.create_mesh_2d(length=2,breadth=5,nelemx=32,nelemy=32,
                          stf='inclusion',bctype='trac',rmin=0.1,rmax=0.2,
                          radius = 1.0,
                          xcen   = 1.0,
                          ycen   = 2.5,
                          stfmin = 11.0,
                          stfmax = 22.0,
                          nu     = 0.25,
                          nclassx=1,
                          nclassy=1
                         )
    mesh2d.write_mesh(filename='data.in')
    mesh2d.json_dump(filename='data.json.in')
    mesh2d.preprocess(suffix='')
    
    # mesh2d2 = FyPyMesh()
    # mesh2d2.json_read(filename='data.in.json')


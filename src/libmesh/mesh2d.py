from fypymesh import *

if __name__ =='__main__':
    mesh2d = FyPyMesh()
    mesh2d.create_mesh_2d(length=1.0,breadth=1.5,nelemx=64,nelemy=96,
                          stftype='inclusion',bctype='trac',
                          radii   = (0.1,0.15,0.05),
                          centers = ((0.25,0.25),(0.75,1.1),(0.25,1.1)),
                          mu      = (2.5,5.0,3.75),
                          muback  = 1.0,
                          nu      = 0.25,
                          eltype  = 'linelas2dnumbasri',
                          bcmag   = -0.06
                         )
    # mesh2d.write_mesh(filename='data.in')
    mesh2d.json_dump(filename='data.json.in')
    mesh2d.preprocess(suffix='_mesh2d')


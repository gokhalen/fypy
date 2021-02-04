from fypymesh import *

if __name__ =='__main__':
    mesh2d = FyPyMesh()
    mesh2d.create_mesh_2d(length=2.0,breadth=2.0,nelemx=2,nelemy=2,
                          stftype='homogeneous',bctype='trac',
                          radii   = [1.0],
                          centers = [[1.0,2.5]],
                          mu      = 2.5,
                          muback  = 1.0,
                          nu      = 0.49,
                          eltype  = 'linelas2dnumbasri'
                         )
    mesh2d.write_mesh(filename='data.in')
    mesh2d.json_dump(filename='data.json.in')
    mesh2d.preprocess(suffix='_mesh2d')


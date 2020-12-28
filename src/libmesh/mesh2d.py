from fypymesh import *

if __name__ =='__main__':
    mesh2d = FyPyMesh()
    mesh2d.create_mesh_2d(length=2,breadth=5,nelemx=32,nelemy=32,
                          stftype='inclusion',bctype='trac',
                          radii   = [1.0],
                          centers = [[1.0,2.5]],
                          mumin   = 11.0,
                          mumax   = 22.0,
                          nu      = 0.4,
                          nclassx=1,
                          nclassy=1
                         )
    mesh2d.write_mesh(filename='data.in')
    mesh2d.json_dump(filename='data.json.in')
    mesh2d.preprocess(suffix='')


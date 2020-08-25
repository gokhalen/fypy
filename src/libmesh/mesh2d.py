from fypymesh import *

if __name__ =='__main__':
    mesh2d = FyPyMesh()
    mesh2d.create_mesh_2d(length=10,breadth=50,nelemx=5,nelemy=3,stf='homogeneous',bctype='trac',filename='data.in')
    mesh2d.write_mesh()

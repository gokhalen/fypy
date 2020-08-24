from fypymesh import *

if __name__ =='__main__':
    mesh2d = FyPyMesh()
    mesh2d.create_mesh_2d(length=10,breadth=50,nelemx=20,nelemy=10,stf='homogeneous',filename='data.in')
    # mesh2d.write_mesh()

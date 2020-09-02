from fypymesh import *

if __name__ =='__main__':
    mesh2d = FyPyMesh()
    mesh2d.create_mesh_2d(length=2,breadth=5,nelemx=10,nelemy=10,stf='homogeneous',bctype='trac')
    mesh2d.write_mesh(filename='data.in')
    mesh2d.json_dump(filename='data.json.in')
    
    # mesh2d2 = FyPyMesh()
    # mesh2d2.json_read(filename='data.in.json')


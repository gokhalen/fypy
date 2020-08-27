from fypymesh import *

if __name__ =='__main__':
    mesh1d = FyPyMesh()
    mesh1d.create_mesh_1d(start=0.0,end=1.0,nelem=10,stf='homogeneous')
    mesh1d.write_mesh(filename='data.in')
    mesh1d.json_dump(filename='data.json.in')

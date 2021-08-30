# fypy: A Python3 based 2D linear elasticity finite element code

THERE IS NO WARRANTY FOR FYPY.

Dependencies: numpy, numba, scipy, json, timerit and multiprocessing, pyvista

To test the code, do the following:

1) Clone FyPy using: git clone https://github.com/gokhalen/fypy.git
2) In the FyPy root directory, type: "cd Problems/2DNew/test_linear_axial_1"
3) Type: python3.8 ../../../main.py --inputfile=data.json.in
4) You should see png files with the results.

For multiprocessing, do the following:
python3.8 ../../../main.py --nprocs=2 --partype=async

# Meshing

1) See fypymesh.py and mesh2d.py in src/libmesh

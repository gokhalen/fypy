# fypy: A Python3 based 2D linear elasticity finite element code

THERE IS NO WARRANTY FOR FYPY.

Dependencies: numpy, numba, scipy, json, timerit and multiprocessing

To test the code, do the following:

1) Clone FyPy using: git clone https://github.com/gokhalen/fypy.git
2) In the FyPy root directory, type: "cd Problems/2D/10X10_Traction_Homegeneous/"
3) Type: python3.8 ../../../main.py data.json.in
4) Run the plot2d.py file in the same directory using Sypder or any IDE with supports plotting

For multiprocessing, do the following:

python3.8 ../../../main.py --nprocs=2 --partype=async


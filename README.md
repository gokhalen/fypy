# fypy: A Python3 based 2D linear elasticity finite element code

Dependencies: Numpy, SciPy and Json, timerit and multiprocessing

To test the code, do the following:

1) In the FyPy root directory, type: "cd Problems/2D/10X10_Traction_Homegeneous/"
2) Type: python3.8 ../../../main.py data.json.in
3) Run the plot2d.py file in the same directory using Sypder or any IDE with supports plotting


Other Notes:
1) To checkout specific tag, first clone the repo and then
      "git checkout <tagname>" e.g. "git checkout v1.0"

2) Tags are added with:
      git tag -a v1.0 -m "Before Parallel Development"

      -a means annotated
      -m means message
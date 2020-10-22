import argparse

def getargs():
    parser = argparse.ArgumentParser(description='FyPy: A finite element code written in Python')
    parser.add_argument('--nprocs',help='number of processes to use',required=False,type=int,default=1)
    parser.add_argument('--chunksize',help='chunksize to use',required=False,type=int,default=1)
    parser.add_argument('--inputfile',help='input json file',required=False,type=str,default='data.json.in')
    parser.add_argument('--outputfile',help='output json file',required=False,type=str,default='data.json.out')
    parser.add_argument('--partype',help='parallelization type: serial poolmap or async',required=False,
                        type=str,default='list',choices=['serial','poolmap','async','list'])
    solverlist = ['spsolve','bicg','bicgstab','cg','cgs','gmres','lgmres','minres','qmr','gcrotmk']
    solverstr  = str(solverlist)
    parser.add_argument('--solvertype',help=f'choose from: {solverstr}',required=False,type=str,default='spsolve',choices=solverlist)
    parser.add_argument('--profile',help=f'runs the assembly through the profiler cProfile',
                        required=False,type=str,default='False',choices=['True','False'])

    args = parser.parse_args()
    return args
    

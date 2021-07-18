# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 10:18:09 2021

@author: aa
"""

# pure shear mesh generation from a axial inputfile created by mesh2d
import json

# open the input file created by mesh2d in libmesh directory
with open('linear.json.in','rt') as fin:
    inputfile_dict= json.load(fin)

nnodex = 65
nnodey = 97
nelem  = (nnodex-1)*(nnodey-1)

# trying to reproduce analytical solution
# ux = (aa)*y, uy=(bb)*x
# solving a pure dirichlet, pure shear problem

# in the original input file need to change nelem,gdofn,conn,ideqn,dirich,trac
# need to change dirich,ideqn,
# make sure that trac is zero everywhere

Lx = 1.0
Ly = 1.5


aa = 0.01
bb = 0.02

deltax = Lx/(nnodex-1)
deltay = Ly/(nnodey-1)

boundarydofn = nnodey*2 + nnodey*2 + (nnodex-2)*2 + (nnodex-2)*2
gdofn        = 2*nnodex*nnodey - boundarydofn

ideqn   = []
dirich  = [[0,0]]*(nnodex*nnodey)
trac    = [[0.0,0.0]]*(nnodex*nnodey)
nodenum = 0
ideqnno = 0 

inode = 0

for inodex in range(nnodex):
    xcoord = inodex*deltax
    for inodey in range(nnodey):
        ycoord = inodey*deltay
        ux = aa*ycoord
        uy = bb*xcoord
        # if node is on the bounary
        if (inodex==0) or (inodey==0) or (inodex==(nnodex-1)) or (inodey==(nnodey-1)):
            dirich[inode]=[ux,uy]
            ideqn.append([-1,-1])
        else:
            ideqn.append([ideqnno,ideqnno+1])
            ideqnno +=2
           
        inode += 1

conn = inputfile_dict['conn']
conn = conn[0:nelem]

inputfile_dict['nelem']=nelem
inputfile_dict['gdofn']=gdofn
inputfile_dict['conn']=conn
inputfile_dict['ideqn']=ideqn
inputfile_dict['dirich']=dirich
inputfile_dict['trac']=trac


        
with open('shear.json.in','wt') as fout:
    json.dump(inputfile_dict,fout,indent=4)        

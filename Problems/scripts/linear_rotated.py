# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 14:38:28 2021

@author: aa
"""

# rotates the problem created by mesh2d in fypymesh
import json

nnodex = 65
nnodey = 97

nelem  = (nnodex-1)*(nnodey-1) + (nnodey-1)

with open('linear.json.in','rt') as fin:
    inputfile_dict=json.load(fin)
    
# need to modify nelem (more traction elements in y-direction)
# gdofn (because nodes in x and y are not equal) 
# ideqn,trac and conn


gdofn = 2*nnodex*nnodey - nnodey - 1

ideqn = []
# set x dofs on left edge
# first node constrained in both x and y direction
ideqnno=0
for inodex in range(nnodex):
    for inodey in range(nnodey):
        # first node
        if ((inodex==0) and (inodey==0)):
            ideqn.append([-1,-1])
        # node on the left which is not first node
        elif (inodex==0):
            ideqn.append([-1,ideqnno])
            ideqnno +=1
        else:
            ideqn.append([ideqnno,ideqnno+1])
            ideqnno +=2

# set trac            
trac = [[0.0,0.0]]*(nnodex*nnodey)
for inode in range(nnodex*nnodey-nnodey+1,nnodex*nnodey+1):
    trac[inode-1] = [-0.06,0.0]
    
# set connectivity
conn = inputfile_dict['conn']
conn = conn[:(nnodex-1)*(nnodey-1)]

for inode in range(nnodex*nnodey-nnodey+1,nnodex*nnodey):
    conn.append([inode,inode+1,'linelastrac2d'])
    
inputfile_dict['nelem']=nelem
inputfile_dict['gdofn']=gdofn
inputfile_dict['ideqn']=ideqn
inputfile_dict['conn']=conn
inputfile_dict['trac']=trac

with open('linear_rotate.json.in','wt') as fout:
    json.dump(inputfile_dict,fout,indent=4)        

    
            
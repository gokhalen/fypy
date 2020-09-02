import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import numpy as np

with open('data.json.out','r') as fin:
    output_data=json.load(fin)
    
with open('data.json.in','r') as fin:
    input_data=json.load(fin)
    
nelemx = input_data['nelemx']
nelemy = input_data['nelemy']
nnodex = nelemx+1
nnodey = nelemy+1
    
coord = input_data['coord']
coord = np.asarray(coord)
coord = coord[:,0:2]
xx    = coord[:,0]
yy    = coord[:,1]


sol  =  output_data['solution']
sol  = np.asarray(sol)
ux   = sol[:,0]
uy   = sol[:,1]

prop = input_data['prop']
prop = np.asarray(prop)
lam  = prop[:,0]
mu   = prop[:,1]

xx  = xx.reshape(nnodex,nnodey)
yy  = yy.reshape(nnodex,nnodey)
ux  = ux.reshape(nnodex,nnodey)
uy  = uy.reshape(nnodex,nnodey)
mu  = mu.reshape(nnodex,nnodey)
lam = lam.reshape(nnodex,nnodey)

plt.figure('lambda')
plt.pcolormesh(xx,yy,lam,vmin=0,vmax=10)
plt.title('lambda')
plt.colorbar()
ax = plt.gca()
ax.set_aspect('equal')

plt.figure('mu')
plt.pcolormesh(xx,yy,mu,vmin=0,vmax=10)
plt.title('mu')
plt.colorbar()
ax = plt.gca()
ax.set_aspect('equal')

plt.figure('ux')
plt.pcolormesh(xx,yy,ux,vmin=-0.2,vmax=0.2)
plt.title('ux')
plt.colorbar()
ax = plt.gca()
ax.set_aspect('equal')

plt.figure('uy')
plt.pcolormesh(xx,yy,uy,vmin=-1.0,vmax=0.1)
plt.title('uy')
plt.colorbar()
ax = plt.gca()
ax.set_aspect('equal')

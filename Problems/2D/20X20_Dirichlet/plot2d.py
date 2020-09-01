import matplotlib.pyplot as plt,json
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

fig, axes = plt.subplots(1,2)

print(axes)

axes[0].pcolormesh(xx,yy,mu)
axes[1].pcolormesh(xx,yy,lam)
# plt.pcolormesh(xx,yy,mu)
# plt.colorbar()
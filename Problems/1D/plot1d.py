import matplotlib.pyplot as plt,json
import numpy as np

with open('data.json.out','r') as fin:
    output_data=json.load(fin)
    
with open('data.json.in','r') as fin:
    input_data=json.load(fin)
    
coord = input_data['coord']
coord = np.asarray(coord)
coord = coord[:,0]

sol   = output_data['solution']
sol   = np.asarray(sol)
sol   = sol[:,0]

plt.plot(coord,sol)
plt.xlabel('x-coordinate')
plt.ylabel('solution') 


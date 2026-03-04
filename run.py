#Imports 

import numpy as np
from math import *
rand = np.random.rand
import matplotlib.pyplot as plt
import SimulationStep as sim

#Number of Particles
p=3
N = 4**p

#Particle Constant Radius and Elasticity
part = {
    "radius" : 0.2,
    "spring" : 250
}


#Timestep
tini = 0
tend=20
h = 0.01
loops = tend/h

t=np.arange(tini,tend,h)

low = np.array([0,0])

upp = np.array([10,10]) * np.sqrt(N)

#Box Dimensions
box = np.vstack([low,upp])

#Gravity Constant
g = 0

#x
x=np.vstack([low[0]+rand(1,N)*(upp[0]-low[0]),
low[1]+rand(1,N)*(upp[1]-low[1])])

#Inital Velocity
vini = 2.5

# v
v=2*(rand(2,N)-0.5)*vini

#Testing functions

# gridArray = gridAssign(x,box,N)
# partNet = connect(gridArray,N)
#

fig, ax = plt.subplots()
scatter  = ax.scatter(x[0],x[1],s=5) 

ax.set_xlim(0,upp[0])
ax.set_ylim(0,upp[0])



for i in range(int(loops)):
    x , v = sim.SimulationStep(x, v, h, part, box, g,N)

    if i%5==0:
        data = np.c_[np.array(x[0]),np.array(x[1])]
        scatter.set_offsets(data)
        plt.pause(0.00001)
plt.show()
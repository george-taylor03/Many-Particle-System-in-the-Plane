#Imports 

import numpy as np
from math import *
# np.random.seed(1)
rand = np.random.rand
import matplotlib.pyplot as plt
import SimulationStep as sim
from matplotlib.patches import Rectangle


#Number of Particles
p=6
N = 4**p

#Particle Constant Radius and Elasticity
part = {
    "radius" : 0.2,
    "spring" : 250
}


#Timestep
tini = 0
tend=100
h = 0.01
loops = tend/h

t=np.arange(tini,tend,h)

#Set up box dimensions
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

#Set up graph
fig, ax = plt.subplots()
scatter  = ax.scatter(x[0],x[1],s=2) 

#Create moving line
boxline = ax.plot([0,upp[0]],[upp[1],upp[1]])[0]

# boxline = ax.plot([0,upp[0]],[low[1],low[1]])[0]


ax.add_patch(Rectangle(low, upp[0] - low[0], upp[1] - low[1], linewidth = 1, fill = False))

diff = [(upp[0] - low[0]) * 0.1, (upp[1] - low[1]) * 0.1]
plt.xlim(low[0] - diff[0], upp[0] + diff[0])
plt.ylim(low[1] - diff[1], upp[1] + diff[1])

#Wall speed
a=100

#Settled Time tor
tor1 = 10
tor2 = tor1 + (5 * np.sqrt(N))/a

torTimes = np.arange(tor1,tor2,h)

print(tor1/h)
print(tor2/h)

for i in range(int(loops)):
    # print(i)
    #Change box dimensions
    if i >= int(tor1/h) and i<=int(tor2/h):
        upp[1] = 10 * np.sqrt(N) - a*(torTimes[i%loops] - tor1)
        # low[1] =  a*(torTimes[i] - tor1)



    #Box Dimensions
    box = np.vstack([low,upp])

    #Run Simulation
    x , v = sim.SimulationStep(x, v, h, part, box, g)

    if i%1==0:
        boxline.set_ydata([upp[1],upp[1]])

        # boxline.set_ydata([low[1],low[1]])
        data = np.c_[np.array(x[0]),np.array(x[1])]
        scatter.set_offsets(data)
        plt.pause(0.00001)
plt.show()
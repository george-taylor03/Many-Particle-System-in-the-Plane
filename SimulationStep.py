#Imports 

import numpy as np
from math import *
np.random.seed(1)
rand = np.random.rand
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#Initial Non-Optimised Version
def forcesCal(x,part):
    Forces = np.zeros((2,N))
    #Collisions made with particle
    for particle in range(N):
        
        #Particle A 
        A = np.array([x[0][particle],x[1][particle]])

        fLeft = max(0,part['radius'] + low[0] - A[0])
        fRight = max(0,part['radius'] + A[0] - upp[0])
        fBot = max(0,part['radius'] + low[1] - A[1])
        fUp = max(0,part['radius'] + A[1] - upp[1])

        #Check Wall Collision
        #X component
        Forces[0][particle] += part['spring'] * (fLeft - fRight)

        #y component
        Forces[1][particle] += part['spring'] * (fBot - fUp)



        #Checking each other particle
        for neighbour in range(N):
            #Other Particle
            B = np.array([x[0][neighbour],x[1][neighbour]])


            #Distance Between them
            d = np.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)

            #Check if will Colide
            if d < 2 * part['radius'] and d>0:
                #Angle
                a = np.arctan2(A[1]-B[1],A[0]-B[0])
            
                #X component
                Forces[0][particle] += part['spring'] * (2*part['radius'] - d) * np.cos(a)

                #y component
                Forces[1][particle] += part['spring'] * (2*part['radius'] - d) * np.sin(a)

    return Forces

def SimulationStep(x, v, h, part, box, g): 

    Forces = forcesCal(x,part)

    xnew = x + (h * v) + (h**2)*Forces
    vnew =(xnew - x)/ h 

    return xnew, vnew


def update(frame):
    # print(frame)
    # print(state[frame][0][0])
    # exit()
    points.set_xdata(state[frame][0][0])
    points.set_ydata(state[frame][0][1])
    return points,

#Number of Particles
p=2
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

# global x
x=np.vstack([low[0]+rand(1,N)*(upp[0]-low[0]),
low[1]+rand(1,N)*(upp[1]-low[1])])

vini = 2.5

# global v
v=2*(rand(2,N)-0.5)*vini

state = []

state.append([x,v,t[0]])
for i in range(int(loops)):

    x , v = SimulationStep(x,v,h,part,box,g)

    state.append([x,v,t[i]])

print(x)
print(v)

#Sets up animated plot
fig, ax = plt.subplots()
points, = ax.plot([],[],'o',ms=3) 

ax.set_xlim(0,upp[0])
ax.set_ylim(0,upp[0])

#Shows animation 
anim = FuncAnimation(fig, update, frames=len(state), blit=True,interval=1)

plt.show()



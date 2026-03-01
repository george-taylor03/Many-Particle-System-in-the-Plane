#Imports 

import numpy as np
from math import *
np.random.seed(1)
rand = np.random.rand
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def gridAssign(x,box,N):

    #Assign gridLength
    gridLength = 5

    #Get number of grid rows and columns
    rows = int(box[1][0] / gridLength)
    cols = int(box[1][0] / gridLength)
    

    #Zero matrix representing grid positions
    gridArray = []

    #Assign Grids
    for particle in range(N):
        #Determine Grid Row

        gridPosRow = int(x[0][particle] // gridLength)

        #Accounting for boundary cases
        if gridPosRow>(rows-1):
            gridPosRow -= 1

        gridPosCol = int(x[1][particle] // gridLength)

        #Accounting for boundary cases
        if gridPosCol>(cols-1):
            gridPosCol -= 1
        gridArray.append([gridPosRow,gridPosCol])

    return gridArray,gridLength

def connect(gridArray,N):
    #Network of particle connections
    partNet=[]
    #Find connections to that particle
    for particle in range(N):
        currNet = []
        #Check of particle connects i.e 1 grid box way
        for neighbour in range(N):
            if abs(gridArray[particle][0] - gridArray[neighbour][0])<=1 and abs(gridArray[particle][1] - gridArray[neighbour][1]) <=1:
                currNet.append(neighbour)
        partNet.append(currNet)
    return partNet

def forceWall(Forces,x,gridLength,part,N):
    #Check collision with wall
    for particle in range(N):
        if x[0][particle] <= gridLength or x[0][particle] >= upp[0] - gridLength or x[1][particle] <= gridLength or x[1][particle] >= upp[1] - gridLength:
            #Particle A 
            A = np.array([x[0][particle],x[1][particle]])

            #Wall Forces
            fLeft = max(0,part['radius'] + low[0] - A[0])
            fRight = max(0,part['radius'] + A[0] - upp[0])
            fBot = max(0,part['radius'] + low[1] - A[1])
            fUp = max(0,part['radius'] + A[1] - upp[1])

            #Check Wall Collision
            #X component
            Forces[0][particle] += part['spring'] * (fLeft - fRight)

            #y component
            Forces[1][particle] += part['spring'] * (fBot - fUp)
    return Forces

def forceParticle(Forces,x,partNet,part):
    #Cycle through each particle
    for particle in range(N):
        #cycle through all its connections
        for neighbour in partNet[particle]:
                    #Particle A 
            A = np.array([x[0][particle],x[1][particle]])

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

def SimulationStep(x, v, h, part, box, g,N): 

    #Set up forces array
    Forces = np.zeros((2,N))

    gridArray,gridLength = gridAssign(x,box,N)

    partNet = connect(gridArray,N)

    Forces = forceWall(Forces,x,gridLength,part,N)

    Forces = forceParticle(Forces,x,partNet,part)

    xnew = x + (h * v) + (h**2)*Forces
    vnew =(xnew - x)/ h 

    return xnew, vnew


#Number of Particles
p=1
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

low = np.array([0,0])

upp = np.array([10,10]) * np.sqrt(N)

#Box Dimensions
box = np.vstack([low,upp])

#Gravity Constant
g = 0

#x
x=np.vstack([low[0]+rand(1,N)*(upp[0]-low[0]),
low[1]+rand(1,N)*(upp[1]-low[1])])

# #Placeholder Positions
# x[0] = [1,10,1,20]
# x[1] = [1,5,11,20]

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
    # box = np.vstack([low,upp-[0,i]])

    x , v = SimulationStep(x, v, h, part, box, g,N)

    if i%10==0:
        data = np.stack([x[0],x[1]]).T
        scatter.set_offsets(data)
        plt.pause(0.001)

plt.show()



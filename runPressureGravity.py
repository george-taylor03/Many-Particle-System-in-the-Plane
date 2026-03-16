
import numpy as np
from math import *
rand = np.random.rand
import matplotlib.pyplot as plt
import SimulationStep as sim
from matplotlib.patches import Rectangle
# Set up values 

# Number of Particles
p=8
N = 4**p

# Maximum p value to be run for part b u
pmax = p

# Particle Constant Radius and Elasticity
part = {
    "radius" : 0.2,
    "spring" : 250
}

# Timestep
tini = 0
tend=40
h = 0.01
loops = int(tend/h)

t=np.arange(tini,tend,h)

low = np.array([0,0])

upp = np.array([10,10]) * np.sqrt(N)

# Box Dimensions
box = np.vstack([low,upp])

# Gravity Constant
g = 0.05

# x
x=np.vstack([low[0]+rand(1,N)*(upp[0]-low[0]),
low[1]+rand(1,N)*(upp[1]-low[1])])

# Inital Velocity
vini = 2.5

# v
v=2*(rand(2,N)-0.5)*vini

#Split into vertical grids
#Number of grids
nbins = 25

binLength = upp[1] / nbins

#y middle of bins for plott
yMids = np.arange(binLength/2,upp[1],binLength)

#Pressure on left wall split into nbins
leftWall = np.zeros(nbins)
#Pressure on right wall split into nbins
rightWall = np.zeros(nbins)

#Forces on left wall split into nbins per iteration
leftForce = np.zeros(nbins)
#Forces on right wall split into nbins per iteration
rightForce = np.zeros(nbins)

#Times to measure pressure 
tor1 = 20
tor2 = 40

#Vary different wall speeds
for i in range(int(loops)):
    #Run Simulation
    x, v, forces_walls, distance, wall_collision_particles, particle_collision_particles, vWalls = sim.SimulationStep(x, v, h, part, box, g)


    if i>= int(tor1/h) and i<int(tor2/h):
        #vWalls is the force for every particle on the vertical wall
        #Note positive force from left wall negative force from right wall
        for particle in range(N):
            if vWalls[particle] !=0:
                #bin location on wall
                loc = int(x[1][particle] / binLength)
                #Left wall
                if vWalls[particle] > 0:
                    leftForce[loc] += vWalls[particle]

                else:
                    rightForce[loc] += -vWalls[particle]
        #Add pressure for this iteration
        leftWall += leftForce / binLength
        rightWall += rightForce / binLength
        #Forces on left wall split into nbins per iteration
        leftForce = np.zeros(nbins)
        #Forces on right wall split into nbins per iteration
        rightForce = np.zeros(nbins)

#Time pressure was taken accross
torDiff = tor2 - tor1

#Average pressure across time 
leftWall /= (torDiff/h)

rightWall /= (torDiff/h)

#Plot pressure on each verticle wall
plt.xlabel("Vertical height y")
plt.ylabel("Pressure")
plt.plot(yMids,leftWall,"o", label = "Pressure of left wall against vertical height")
plt.legend()
plt.show()

plt.xlabel("Vertical height y")
plt.ylabel("Pressure")
plt.plot(yMids,rightWall,"o", label = "Pressure of right wall against vertical height")
plt.legend
plt.show()
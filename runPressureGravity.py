import numpy as np
import matplotlib.pyplot as plt
import math
from SimulationStep import SimulationStep
from numba import njit, prange, get_num_threads, get_thread_id
rand = np.random.rand

# Note: For larger bins - might be faster to make parallel lists 2D (thread_num x nbins)

# Adds pressure from current iteration to total pressure
@njit(parallel = True)
def calculate_pressure(N, nbins, binLength, Y, vWalls, leftWall, rightWall):
    thread_num = get_num_threads()
    # To remove race conditions each thread has it's own list
    leftForceParallel = np.zeros(nbins * thread_num)
    rightForceParallel = np.zeros(nbins * thread_num)

    for particle in prange(N):
        # Force for current particle against vertical wall
        vWallsParticle = vWalls[particle]
        
        if vWallsParticle != 0:
            # Bin location on wall
            loc = int(Y[particle] / binLength)
            
            # vWallsParticle is positive if particle is at left wall and negative if at right wall
            # Left wall
            if vWallsParticle > 0:
                leftForceParallel[loc + nbins * get_thread_id()] += vWallsParticle
            # Right wall
            else:
                rightForceParallel[loc + nbins * get_thread_id()] -= vWallsParticle

    # Converts parallel lists to normal lists
    leftForce = np.zeros(nbins)
    rightForce = np.zeros(nbins)

    for i in range(thread_num):
        thread_list = i * nbins
        for j in range(nbins):
            leftForce[j] += leftForceParallel[thread_list + j]
            rightForce[j] += rightForceParallel[thread_list + j]
    
    # Add pressure from current iteration
    leftWall += leftForce / binLength
    rightWall += rightForce / binLength

# Number of Particles
p = 8
N = 4 ** p

# Particle constant radius and elasticity
part = {"radius" : 0.2, "spring" : 250}

# Timestep
tend = 40
h = 0.01
loops = int(tend / h)

low = np.array([0, 0])
upp = np.array([10, 10]) * math.sqrt(N)

# Box Dimensions
box = np.vstack([low, upp])

# Gravity
g = 0.05

# Initial Position
x = np.vstack([low[0] + rand(1, N) * (upp[0] - low[0]),
              low[1] + rand(1, N) * (upp[1] - low[1])])

# Inital Velocity
vini = 2.5
v = 2 * (rand(2, N) - 0.5) * vini

# Split box into vertical grids
# Number of grids
nbins = 25

# Length of each grid
binLength = upp[1] / nbins

# y middle of bins for plot
yMids = np.arange(binLength / 2, upp[1], binLength)

# Pressure on left and right wall
leftWall = np.zeros(nbins)
rightWall = np.zeros(nbins)

# Times to measure pressure 
tor1 = 20
tor2 = 40

# Used in simulation for times in [tor1, tor2]
condition1, condition2 = int(tor1 / h), int(tor2 / h) # Note: Come up with better variable names

# Simulation
for i in range(loops):
    x, v, _, _, _, _, vWalls = SimulationStep(x, v, h, part, box, g)

    if condition1 <= i < condition2: # <= int(tor2 / h)?
        calculate_pressure(N, nbins, binLength, x[1], vWalls, leftWall, rightWall)

# Time pressure was taken accross
torDiff = tor2 - tor1

# Average pressure across time 
leftWall *= h / torDiff
rightWall *= h / torDiff

# Plot pressure on each verticle wall
plt.xlabel("Vertical height y")
plt.ylabel("Pressure")
plt.plot(yMids, leftWall, "o", label = "Pressure of left wall against vertical height")
plt.legend(loc = 'upper right')
plt.show()

plt.xlabel("Vertical height y")
plt.ylabel("Pressure")
plt.plot(yMids, rightWall, "o", label = "Pressure of right wall against vertical height")
plt.legend(loc = 'upper right')
plt.show()

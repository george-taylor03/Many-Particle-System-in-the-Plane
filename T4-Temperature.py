import numpy as np
import matplotlib.pyplot as plt
import math
from SimulationStep import SimulationStep
from numba import njit

rand = np.random.rand

# Adds temperatures for each particle from current iteration to total temperature (and counter)
@njit
def calculate_temperature(N, nbins, binLength, Y, vx, vy, temp, count):
    for particle in range(N):
        # Velocity of particle
        particle_vx, particle_vy = vx[particle], vy[particle]
        
        # Temperature of particle
        particle_temp = (particle_vx * particle_vx + particle_vy * particle_vy) * 0.5
        
        # Bin location
        loc = int(Y[particle] / binLength)

        if nbins <= loc:
            loc = nbins - 1
        elif loc < 0:
            loc = 0
        
        # Adding temperature of particle to temp
        temp[loc] += particle_temp
        count[loc] += 1
        
# Number of Particles
p = 8
N = 4 ** p

# Particle constant radius and elasticity
part = {"radius" : 0.2, "spring" : 250}

# Gravity
g = 0.05

# Timestep
tend = 40
h = 0.01
loops = int(tend / h)

low = np.array([0, 0])
upp = np.array([10, 10]) * math.sqrt(N)

# Box Dimensions
box = np.vstack([low, upp])

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

# List of temperatures for each bin
temp = np.zeros(nbins)

# List of number of particles for each bin
count = np.zeros(nbins)

# Times to measure pressure 
tor1 = 20
tor2 = 40

# Used in simulation for times in [tor1, tor2]
tor1_condition, tor2_condition = int(tor1 / h), int(tor2 / h)

# Simulation
for i in range(loops):
    x, v, *_ = SimulationStep(x, v, h, part, box, g)

    if tor1_condition <= i <= tor2_condition:
        calculate_temperature(N, nbins, binLength, x[1], v[0], v[1], temp, count)

# Averaging temperatures - https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero
avg_temp = np.divide(temp, count, out = np.zeros_like(temp), where = count != 0)

# Plotting yMids against average temperatures
plt.xlabel("Vertical height y")
plt.ylabel("Average temperature")
plt.plot(yMids, avg_temp, "o", label = "Average temperature in bin")
plt.legend(loc = 'upper right')
plt.show()
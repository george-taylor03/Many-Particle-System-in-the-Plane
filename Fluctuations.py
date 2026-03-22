import numpy as np
import matplotlib.pyplot as plt
import math
from SimulationStep import SimulationStep

rand = np.random.rand

# Average temperature
def AvgTemp(v):
    return np.sum(v * v) / (2 * v.shape[1])
    
# Standard deviation of temperature for the last nrec timesteps
def sigma(T):
    return np.std(T[-nrec:])

# Set up conditions to run the different experiments for T2
# Change to true or false depending on which you want to run
# Can all be true at once 

# For part a 
partA = False
# For temperature graphs at each value of p (part a)
temperatureGraph = False

# For part b
partB = False

# For part c 
partC = True 

# Number of Particles
p = 4
N = 4 ** p

# Maximum p value to be run for part b
pmax = p

# Particle constant radius and elasticity
part = {"radius" : 0.2, "spring" : 250}

# Gravity
g = 0

# Timestep (starts at t = 0)
tend = 10
h = 0.01
loops = int(tend / h)

# Last number of timesteps to be analysed
nrec = 500

low = np.array([0, 0])
upp = np.array([10, 10]) * math.sqrt(N)

# Box Dimensions
box = np.vstack([low, upp])

# Initial position
x = np.vstack([low[0] + rand(1, N) * (upp[0] - low[0]),
              low[1] + rand(1, N) * (upp[1] - low[1])])

# Inital Velocity
vini = 2.5
v = 2 * (rand(2, N) - 0.5) * vini


"""" loglog for part a """
if partA:
    # Average temperatures
    T = np.zeros(loops)
    # Standard deviations
    S = np.zeros(pmax)

    # List of N's for plotting
    Nplot = []

    # Collecting data for each p up to pmax
    for p in range(1, pmax + 1):
        # Number of Particles
        N = 4 ** p

        # N values for plotting
        Nplot.append(N)
    
        low = np.array([0, 0])
        upp = np.array([10, 10]) * math.sqrt(N)
    
        #Box Dimensions
        box = np.vstack([low, upp])
        
        # Initial position
        x = np.vstack([low[0] + rand(1, N) * (upp[0] - low[0]),
                       low[1] + rand(1, N) * (upp[1] - low[1])])

        # Inital Velocity
        vini = 2.5
        v = 2 * (rand(2, N) - 0.5) * vini
        
        for i in range(loops):
            x, v = SimulationStep(x, v, h, part, box, g)[0 : 2]
            
            T[i] = AvgTemp(v)
        
        S[p - 1] = sigma(T)
        
        nrecTemp = T[-nrec:]
        nrecTempMean = nrecTemp.mean()
        print(f"Over the last {nrec} timesteps, average temperature = {nrecTempMean} and range as a fraction of the average = {(nrecTemp.max() - nrecTemp.min()) / nrecTempMean}\n")
        
        # Temperature graphs for each p
        if temperatureGraph:
            plt.plot(np.arange(0, loops, 1), T, label = 4 ** p)
            plt.xlabel('Timestep')
            plt.ylabel('Temperature')
            plt.legend(loc = 'lower left')
            plt.show()

    # Plotting N versus standard deviation on a loglog scale
    plt.loglog(Nplot, S, 'o-' )
    plt.xlabel('Number of particles')
    plt.ylabel('Standard deviation')
    plt.show()


""" histogram for part b """
if partB:
    for i in range(loops):
        x, v = SimulationStep(x, v, h, part, box, g)[0 : 2]

    speed = np.sqrt(np.sum(v ** 2, 0))
    
    # number of bins not sure what final value will be 
    Bins = 30
    plt.hist(speed, bins = Bins, density = True, edgecolor = 'k')
    plt.ylabel('Number of particles')
    plt.xlabel('Final speed value')
    plt.show()


""" mean free path for part c """
if partC:
    # Total distance travelled for each particle
    distance = np.zeros(N)

    # Particles that collided with walls in previous time step (True = colliding)
    old_wall_collision_particles = np.zeros(N, np.bool_)

    # Pairs of particles that collided in previous time step
    old_particle_collision_particles = set()

    # Count for each particle's number of collisions
    particle_collision_count = np.ones(N)
    
    for i in range(loops):
        x, v, _, distance_step, new_wall_collision_particles, new_particle_collision_particles, _ = SimulationStep(x, v, h, part, box, g)

        distance += distance_step

        # Adds 1 to particle's collision count if particle is in the new collision list but not in the old list
        # Sets old collision lists to new after counting
        
        # Wall collisions
        for particle in range(N):
            if new_wall_collision_particles[particle] and not old_wall_collision_particles[particle]:
                particle_collision_count[particle] += 1
        
        old_wall_collision_particles = new_wall_collision_particles

        # Particle collisions
        if new_particle_collision_particles:
            for particle, neighbour in new_particle_collision_particles - old_particle_collision_particles:
                particle_collision_count[particle] += 1
                particle_collision_count[neighbour] += 1

            old_particle_collision_particles = new_particle_collision_particles
    
    # Overall mean free path
    print(f'Overall mean free path: {np.average(distance / particle_collision_count)}')

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Rectangle

# Temp lines from experimenting, final solution probably won't need any other than njit and prange
from numba import njit, int64, prange
from numba.typed import List, Dict
from numba.core import types


###

# Note: solution uses Numba, if unfamiliar - https://numba.readthedocs.io/en/stable/user/index.html (Don't have to read)
# I haven't fully read doc so might be missing some improvements

# TO DO (potential improvements):
    # Find a way for grid to not be a dictionary without making it a numba dict - Grid ideally becomes array / list, new grid should act the same or similar
        # After that, can use @njit(...) on force_particle -> will be faster
    # Find way to reduce math.function (e.g. math.cos) in force_particle, ideally get rid of all -> sqrt not as expensive as others
    # Go through Performance Improvements (Link, User Manual -> Performance Tips) on njit functions at the end, see if time goes down

# Improvements aren't really necessary its pretty fast as is, just ideas

# Numba stuff might make writing solutions for other parts a bit annoying, can comment out njits and it's still quite fast.

###

#np.random.seed(1)
ran = np.random.rand

@njit
def grid_index(N, box_length, length, width, X, Y):
    grid_index = []
     # Assigns each particle to list in grid
    for particle in range(N):
        particle_X, particle_Y = X[particle], Y[particle]
        
        particle_col, particle_row = int(particle_X / box_length), int(particle_Y / box_length)

        if particle_col > length - 1:
            particle_col -= 1
        
        if particle_row > width - 1:
            particle_row -= 1

        grid_index.append([particle_col, particle_row])

    return grid_index

# Completing To Do list will probably allow @njit(...) to be used on function? Might make a small difference
def grid_builder(N, grid_index):
    grid = {}

    # Assigns each particle to list in grid
    for particle in range(N):
        particle_col, particle_row = grid_index[particle]
        
        if (particle_col, particle_row) in grid:
            grid[particle_col, particle_row].append(particle)
        else:
            grid[particle_col, particle_row] = [particle]

    return grid

# Checks for collisions with particles
#@njit(fastmath = True) -> Improvement when working
def force_particle(N, radius, spring, X, Y, grid_index_, forces, grid):
    for particle in range(N):
        # Index of particle in grid_index
        particle_index_X, particle_index_Y = grid_index_[particle][0], grid_index_[particle][1]

        # Checks boxes around particle's box
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                current_box = (particle_index_X + i, particle_index_Y + j)
                if current_box in grid:
                    # Calculates forces between current particle and all other particles in current_box
                    for neighbour in grid[current_box]:
                        # Ignore checking against itself
                        if neighbour != particle:
                            # Positions of particle and neighbour
                            particle_X, particle_Y = X[particle], Y[particle] 
                            neighbour_X, neighbour_Y = X[neighbour], Y[neighbour]
                            
                            # Distance between the two particles (squared)
                            d = (particle_X - neighbour_X) ** 2 + (particle_Y - neighbour_Y) ** 2
            
                            # Checks if particles have collided
                            if 0 < d < (2 * radius) ** 2:
                                # Actual distance
                                d = math.sqrt(d)
                
                                # Angle between particles
                                a = np.arctan2(particle_Y - neighbour_Y, particle_X - neighbour_X)

                                coeff = spring * (2 * radius - d)
                
                                forces[0, particle] += coeff * math.cos(a)
                                forces[1, particle] += coeff * math.sin(a)

# Checks for collisions with wall
@njit(fastmath = True) # Note: parallel=True, sometimes faster -> maybe faster with larger n
def force_wall(N, radius, spring, X, Y, box, forces):
    for particle in prange(N):
        particle_X, particle_Y = X[particle], Y[particle]

        # Only calculate collisions for particles less than radius away from a wall
        if particle_X < radius or particle_Y < radius or particle_X > box[1, 0] - radius or particle_Y > box[1, 1] - radius:

            f_left = max(0, radius - particle_X)
            f_right = max(0, radius + particle_X - box[1, 0])
            f_bottom = max(0, radius - particle_Y)
            f_up = max(0, radius + particle_Y - box[1, 1])
            
            forces[0, particle] += spring * (f_left - f_right)
            forces[1, particle] += spring * (f_bottom - f_up)

def simulation_step(N, dt, radius, spring, gravity, box_length, length, width, x, v, box):
    X, Y = x[0], x[1]

    # Some variable names have extra _, just to separate variables from functions, temporary
    grid_index_ = grid_index(N, box_length, length, width, X, Y)
    grid = grid_builder(N, grid_index_)

    forces = np.zeros((2, N))
    
    force_wall(N, radius, spring, X, Y, box, forces)
    force_particle(N, radius, spring, X, Y, grid_index_, forces, grid)
    
    x_new = x + dt * v + dt ** 2 * forces # Updated positions
    v_new = (x_new - x) / dt # Updated velocities
    
    return x_new, v_new


# Number of Particles
p = 4
N = 4 ** p

radius = 0.2 # Radius of particles
spring = 250 # Spring constant of particles
gravity = 0

# Box
low = np.array([0, 0])
upp = np.array([10, 10]) * np.sqrt(N)
box = np.vstack([low, upp])


# Initial position of particles
x = np.vstack([low[0] + ran(1, N) * (upp[0] - low[0]),
               low[1] + ran(1, N) * (upp[1] - low[1])])

# Initial velocities of particles
v_ini = 2.5
v = 2 * (ran(2, N) - 0.5) * v_ini

dt = 0.01 # Timestep
end_time = 10

# Size of each box in grid in simulation_step
box_length = 2 * radius

# Used for grid in simulation_step
length = int(math.ceil((upp[0] - low[0]) / box_length))
width = int(math.ceil((upp[1] - low[1]) / box_length))

# Simulation + Plotting
fig, ax = plt.subplots()

ax.add_patch(Rectangle(low, upp[0] - low[0], upp[1] - low[1], linewidth = 1, fill = False))

diff = [(upp[0] - low[0]) * 0.1, (upp[1] - low[1]) * 0.1]
plt.xlim(low[0] - diff[0], upp[0] + diff[0])
plt.ylim(low[1] - diff[1], upp[1] + diff[1])

scatter = ax.scatter(x[0], x[1], s = radius * 10)

import time # For measuring average time of simulation_step, temporary
times = []

for i in range(int(end_time / dt)):
    s = time.perf_counter()
    x, v = simulation_step(N, dt, radius, spring, gravity, box_length, length, width, x, v, box) # Change inputs later to match SimulationStep in instructions
    e = time.perf_counter()
    times.append(e - s)
    
    #scatter.set_offsets(np.c_[x[0], x[1]])
    #plt.pause(0.01)

times.pop(0)
print(np.average(times))
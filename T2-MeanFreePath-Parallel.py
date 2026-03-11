import numpy as np
import matplotlib.pyplot as plt
import math
import time
from matplotlib.patches import Rectangle
from numba import njit, prange, int64, boolean, get_num_threads, get_thread_id
from numba.typed import List, Dict
from numba.core import types

# Note: Faster than T2-MeanFreePath for around p > 5

# Changes: force_particle is parallel

# Small explanation:
    # if force_particle is parallel -> method of finding particle collisions changes
    
    # parallel = True for function -> each available CPU core carries out function separately
    # Dictionaries aren't thread safe -> normal dictionary can't be used in parallel function
    # Instead create list of dictionaries (length = number of available CPU cores)
    # Each thread adds to their own dictionary in list -> get's rid of thread safety problem
    # Add each key in each dictionary in list to set (= particle_collision_particles)

np.random.seed(1)
ran = np.random.rand

# Creates list of all particle's indexes
@njit
def create_grid_index(N, box_length, length, width, X, Y):
    grid_index = np.empty((2, N), dtype = int64)
    
    for particle in range(N):
        # Position of partcle
        particle_X, particle_Y = X[particle], Y[particle]
        
        # Prevents particle's index from being outside the grid
        particle_col, particle_row = min(int(particle_X / box_length), length - 1), min(int(particle_Y / box_length), width - 1)
        
        grid_index[0, particle] = particle_col
        grid_index[1, particle] = particle_row

    return grid_index

# Returns grid of all particles
@njit
def create_grid(N, length, grid_index, grid):
    for particle in range(N):
        # Particle's index
        particle_col, particle_row = grid_index[0, particle], grid_index[1, particle]

        # Converts (row, col) into integer - https://stackoverflow.com/questions/1730961/convert-a-2d-array-index-into-a-1d-index
        key = int64(particle_row * length + particle_col)
        
        if key not in grid:
            grid[key] = List.empty_list(int64)
        
        grid[key].append(particle)
        
    return grid

# Checks for collisions with walls
# Returns indexes of all particles colliding with wall at current step
@njit(fastmath = True, parallel = True)
def force_wall(N, radius, spring, X, Y, box, forces):
    # Note: If wall forces are changed to stop them from going through walls for multiple steps -> can instead create collision count for all particles and add 1 whenever wall collision
    wall_collision_particles = np.zeros(N, dtype = np.bool_)
    
    for particle in prange(N):
        # Position of partcle
        particle_X, particle_Y = X[particle], Y[particle]

        # Only calculate collisions for particles less than radius away from a wall
        if particle_X < radius or particle_Y < radius or box[1, 0] - radius < particle_X or box[1, 1] - radius < particle_Y:

            f_left = max(0, radius - particle_X)
            f_right = max(0, radius + particle_X - box[1, 0])
            f_bottom = max(0, radius - particle_Y)
            f_up = max(0, radius + particle_Y - box[1, 1])
            
            forces[0, particle] += spring * (f_left - f_right)
            forces[1, particle] += spring * (f_bottom - f_up)           

            wall_collision_particles[particle] = True

    return wall_collision_particles

# Checks for collisions with particles
@njit(fastmath = True, parallel = True)
def force_particle(N, radius, spring, width, X, Y, grid_index, forces, grid, particle_dicts):
    # Condition used to check if two particles have collided
    condition = 4 * radius * radius # = (2 * radius) ** 2
    
    for particle in prange(N):
        # Index of particle in grid_index
        particle_index_X, particle_index_Y = grid_index[0, particle], grid_index[1, particle]
        
        # Force to be added for particle
        particle_force_X, particle_force_Y = 0, 0
        
        # Position of particle
        particle_X, particle_Y = X[particle], Y[particle]
        
        # Checks boxes around particle's box
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                neighbour_index = (particle_index_X + i) + (particle_index_Y + j) * width
                if neighbour_index in grid:
                    # Calculates forces between current particle and all other particles in current_box
                    for neighbour in grid[neighbour_index]:
                        # Uses Fa = -Fb
                        if particle < neighbour:
                            # Positions of neighbour
                            neighbour_X, neighbour_Y = X[neighbour], Y[neighbour]
                            
                            # Distance between the two particles (squared)
                            diff_X = particle_X - neighbour_X
                            diff_Y = particle_Y - neighbour_Y
                            distance = diff_X * diff_X + diff_Y * diff_Y
            
                            # Checks if particles have collided
                            if 0 < distance < condition:
                                # Actual distance
                                distance = math.sqrt(distance)
                
                                # Angle between particles
                                alpha = math.atan2(diff_Y, diff_X)

                                # Forces
                                coeff = spring * (2 * radius - distance)

                                force_X, force_Y = coeff * math.cos(alpha), coeff * math.sin(alpha)
                
                                particle_force_X += force_X
                                particle_force_Y += force_Y
                                
                                forces[0, neighbour] -= force_X
                                forces[1, neighbour] -= force_Y

                                particle_dicts[get_thread_id()][(int64(particle), neighbour)] = True

        forces[0, particle] += particle_force_X
        forces[1, particle] += particle_force_Y

# Returns single time step
def simulation_step(N, dt, radius, spring, gravity, box_length, length, width, x, v, box):
    # Key = index of grid (int), Value = indexes of particles (list)
    grid_empty = Dict.empty(key_type = types.int64, value_type = types.ListType(types.int64))
    
    # x and y coordinates of all particles
    X, Y = x[0], x[1]

    # Splits box into grid and assigns each particle to grid
    grid_index = create_grid_index(N, box_length, length, width, X, Y)
    grid = create_grid(N, length, grid_index, grid_empty)

    # Forces of all particles
    forces = np.zeros((2, N))

    # Wall forces
    wall_collision_particles = force_wall(N, radius, spring, X, Y, box, forces)

    # List of dictionaries, length is number of CPU cores available
    particle_dicts = []
    
    for i in range(get_num_threads()):
        # value doesn't matter for dictionaries, only need keys (= (particle index, neighbour index))
        particle_dicts.append(Dict.empty(key_type = types.UniTuple(types.int64, 2), value_type = types.boolean))

    # Particle forces
    force_particle(N, radius, spring, width, X, Y, grid_index, forces, grid, particle_dicts)

    # Merging all dictionaries in particle_dicts into one set
    particle_collision_particles = set()
    
    for dict in particle_dicts:
        for key in dict:
            particle_collision_particles.add(key)
    
    x_new = x + dt * v + dt * dt * forces # Updated positions
    x_diff = x_new - x # Difference between old and new positions
    v_new = x_diff / dt # Updated velocities

    X_diff, Y_diff = x_diff[0], x_diff[1]
    distance = np.sqrt(X_diff * X_diff + Y_diff * Y_diff) # Distance travelled in step

    return x_new, v_new, distance, wall_collision_particles, particle_collision_particles

# Number of Particles
p = 3
N = 4 ** p

radius = 0.2 # Radius of particles
spring = 250 # Spring constant of particles
gravity = 0 # Gravity

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

# Size of each box in grid
box_length = 2 * radius

# Number of smaller boxes in box's length and width
length = int(math.ceil((upp[0] - low[0]) / box_length))
width = int(math.ceil((upp[1] - low[1]) / box_length))

# Total distance travelled for each particle
distance = np.zeros((2, N))

# Particles that collided with walls in previous time step
old_wall_collision_particles = np.zeros(N, np.bool_)

# Pairs of particles that collided in previous time step
old_particle_collision_particles = set()

# Count for each particle's number of collisions
particle_collision_count = np.ones(N)

# List of times
times = []

# Simulation + Plotting
fig, ax = plt.subplots()

ax.add_patch(Rectangle(low, upp[0] - low[0], upp[1] - low[1], linewidth = 1, fill = False))

diff = [(upp[0] - low[0]) * 0.1, (upp[1] - low[1]) * 0.1]
plt.xlim(low[0] - diff[0], upp[0] + diff[0])
plt.ylim(low[1] - diff[1], upp[1] + diff[1])

#scatter = ax.scatter(x[0], x[1], s = radius * 10)

for i in range(int(end_time / dt)):
    s = time.perf_counter()

    x, v, distance_step, new_wall_collision_particles, new_particle_collision_particles = simulation_step(N, dt, radius, spring, gravity, box_length, length, width, x, v, box)

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
        for particle, neighbour in new_particle_collision_particles:
            if (particle, neighbour) not in old_particle_collision_particles:
                particle_collision_count[particle] += 1
                particle_collision_count[neighbour] += 1

        old_particle_collision_particles = new_particle_collision_particles
    
    e = time.perf_counter()
    times.append(e - s)

    #scatter.set_offsets(np.c_[x[0], x[1]])
    #plt.pause(0.01)

# Average simulation_step runtime
times.pop(0)
print(f'Average simulation_step runtime for {N} (4 ^ {p}) particles: {np.average(times)}')

# Overall mean free path
print(f'Overall mean free path: {np.average(distance / particle_collision_count)}')
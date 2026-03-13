import numpy as np
import matplotlib.pyplot as plt
import math
import time
from matplotlib.patches import Rectangle
from numba import njit, prange, int64, get_num_threads, get_thread_id #--Numba Ver--#
from numba.typed import List, Dict #--Numba Ver--#
from numba.core import types #--Numba Ver--#

#~---------------------------------------------------------------------------------------------------------------------------------------------------~#

# Numba Notes:
    # User manual - https://numba.readthedocs.io/en/stable/user/index.html
    # Lines with comments '#--Numba Ver--#' -> uncommenting = Numba version
    # Lines with comments '#--Python Ver--#' -> uncommenting = Python version
        # Choose one version to run
    # If creating code in Python first -> avoid dictionaries where possible (initializing a numba.typed.dict (like in simulation_step) can be slow)

# Potential Improvements:
    # create_grid_index and create_grid can be merged -> might make a small change, better readability when separate (in my opinion)
    # Find way to reduce math.function (e.g. math.cos) in force_particle, ideally get rid of all -> sqrt not as expensive as others
    # Go through Performance Improvements (Link, User Manual -> Performance Tips) on njit functions, see if time goes down
        # Already done: fastmath, parallel

# Small optimisation notes:
    # x * x faster than x ** 2
    # math.function faster for numbers, np.function faster for arrays
    # array[:, n] slower than array[0, n], array[1, n], ...
    # Splitting x into X (=x[0]) and Y (=x[1]) usually faster

#~---------------------------------------------------------------------------------------------------------------------------------------------------~#

np.random.seed(1)
ran = np.random.rand

# Creates list of all particle's indexes
@njit #--Numba Ver--#
def create_grid_index(N, box_length, length, width, X, Y):
    grid_index = np.empty((2, N), dtype = int64) #--Numba Ver--#
    #grid_index = np.empty((2, N)) #--Python Ver--#
    
    for particle in range(N):
        # Position of partcle
        particle_X, particle_Y = X[particle], Y[particle]
        
        # Prevents particle's index from being outside the grid
        particle_col, particle_row = min(int(particle_X / box_length), length - 1), min(int(particle_Y / box_length), width - 1)
        
        grid_index[0, particle] = particle_col
        grid_index[1, particle] = particle_row

    return grid_index

# Returns grid of all particles
@njit #--Numba Ver--#
def create_grid(N, length, grid_index, grid):
    for particle in range(N):
        # Particle's index
        particle_col, particle_row = grid_index[0, particle], grid_index[1, particle]

        # Converts (row, col) into integer - https://stackoverflow.com/questions/1730961/convert-a-2d-array-index-into-a-1d-index
        key = int64(particle_row * length + particle_col) #--Numba Ver--#
        #key = particle_row * length + particle_col #--Python Ver--#
        
        if key not in grid:
            grid[key] = List.empty_list(int64) # --Numba Ver--
            #grid[key] = [] #--Python Ver--#
        
        grid[key].append(particle)
        
    return grid

# Checks for collisions with walls
@njit(fastmath = True, parallel = True) #--Numba Ver--# 
def force_wall(N, radius, spring, X, Y, box, forces):
     # Note: If wall forces are changed to stop them from going through walls for multiple steps -> can instead create collision count for all particles and add 1 whenever wall collision
    wall_collision_particles = np.zeros(N, dtype = np.bool_)
    #Array of forces  per wall
    forceWalls = np.zeros(4)

    for particle in prange(N): #--Numba Ver--#
    #for particle in range(N): #--Python Ver--#
        # Position of partcle
        particle_X, particle_Y = X[particle], Y[particle]


        # Only calculate collisions for particles less than radius away from a wall
        if particle_X < radius + box[0,0] or particle_Y < radius + box[0,1] or box[1, 0] - radius < particle_X or box[1, 1] - radius < particle_Y:

            # Position of partcle
            particle_X, particle_Y = X[particle], Y[particle]

            #Assign wall focres
            f_left = max(0, radius + box[0,0] - particle_X)
            f_right = max(0, radius + particle_X - box[1, 0])
            f_bottom = max(0, radius + box[0,1] - particle_Y)
            f_up = max(0, radius + particle_Y - box[1, 1])

            #Adds up forces per wall
            forceWalls[0] += f_left
            forceWalls[1] += f_right
            forceWalls[2] += f_bottom
            forceWalls[3] += f_up
            
            forces[0, particle] += spring * (f_left - f_right)
            forces[1, particle] += spring * (f_bottom - f_up)
            
            wall_collision_particles[particle] = True
    
    return forceWalls,forces,wall_collision_particles

# Checks for collisions with particles
@njit(fastmath = True, parallel = True) #--Numba Ver--#
def force_particle(N, radius,spring, width, X, Y, grid_index, forces, grid,particle_dicts):

    # Condition used to check if two particles have collided
    condition = 4 * radius * radius # = (2 * radius) ** 2
    
    for particle in prange(N): #--Numba Ver--#
    #for particle in range(N): #--Python Ver--#
        # Index of particle in grid_index
        particle_index_X, particle_index_Y = grid_index[0, particle], grid_index[1, particle]
        
        # Force to be added for particle
        particle_force_X, particle_force_Y = 0, 0
        
        # Checks boxes around particle's box
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                neighbour_index = (particle_index_X + i) + (particle_index_Y + j) * width
                if neighbour_index in grid:
                    # Calculates forces between current particle and all other particles in current_box
                    for neighbour in grid[neighbour_index]:
                        # Positions of particle and neighbour
                        particle_X, particle_Y = X[particle], Y[particle] 
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

    return forces

# Returns single time step
def SimulationStep(x,v,dt,part,box,g):
    #Assings radius and spring constant
    radius = part["radius"]
    spring = part["spring"]

    #Assign N
    N = len(x[0])
    
    # Size of each box in grid
    box_length = 2 * radius

    # Used for grid
    length = int(math.ceil((box[1][0] - box[0][0]) / box_length))
    width = int(math.ceil((box[1][1] - box[0][1]) / box_length))

    # Key = index of grid (int), Value = indexes of particles (list)
    grid_empty = Dict.empty(key_type = types.int64, value_type = types.ListType(types.int64)) #--Numba Ver--#
    #grid_empty = {} #--Python Ver--#
    # List of dictionaries, length is number of CPU cores available
    particle_dicts = []
    
    for i in range(get_num_threads()):
        # value doesn't matter for dictionaries, only need keys (= (particle index, neighbour index))
        particle_dicts.append(Dict.empty(key_type = types.UniTuple(types.int64, 2), value_type = types.boolean))
    
    # x and y coordinates of all particles
    X, Y = x[0], x[1]

    # Splits box into grid and assigns each particle to grid
    grid_index = create_grid_index(N, box_length, length, width, X, Y)
    grid = create_grid(N, width, grid_index, grid_empty)

    # Forces of all particles
    forces = np.zeros((2, N))
    
    forcesWalls,forces,wall_collision_particles = force_wall(N,radius,spring, X, Y, box, forces)
    forces = force_particle(N, radius,spring, width, X, Y, grid_index, forces, grid,particle_dicts)
    forces[1, :] += -g
    particle_collision_particles = set()
    
    for dict in particle_dicts:
        for key in dict:
            particle_collision_particles.add(key)
    
    
    x_new = x + dt * v + dt * dt * forces # Updated positions
    ######
    x_diff = x_new - x # Difference between old and new positions
    v_new = x_diff / dt # Updated velocities

    X_diff, Y_diff = x_diff[0], x_diff[1]
    distance = np.sqrt(X_diff * X_diff + Y_diff * Y_diff) # Distance travelled
    return x_new, v_new , forcesWalls, distance, wall_collision_particles, particle_collision_particles

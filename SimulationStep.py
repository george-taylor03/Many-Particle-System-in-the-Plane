import numpy as np
import math
from numba import njit, prange, int64, set_num_threads, get_num_threads, get_thread_id
from numba.typed import List, Dict
from numba.core import types

# Numba: User manual - https://numba.readthedocs.io/en/stable/user/index.html

set_num_threads(4)

# Creates grid with all particle's indexes
# Returns grid (dict) + cell index of particles (array)
@njit
def create_grid(N, cell_length, length, X, Y, grid):
    cell_index = np.zeros((2, N), dtype = np.int64)
    
    cell_length_inv = 1 / cell_length
    
    for particle in range(N):
        # Position of particle
        particle_X, particle_Y = X[particle], Y[particle]
        
        # Particle's cell index
        particle_index_X, particle_index_Y = int(particle_X * cell_length_inv), int(particle_Y * cell_length_inv)

        cell_index[0, particle] = particle_index_X
        cell_index[1, particle] = particle_index_Y
        
        # Converts cell index into integer - https://stackoverflow.com/questions/1730961/convert-a-2d-array-index-into-a-1d-index
        key = particle_index_Y * length + particle_index_X

        if key not in grid:
            grid[key] = List.empty_list(int64)
        
        grid[key].append(particle)

    return grid, cell_index

# Calculates forces for all particles
# Returns forces (array) + forces on vertical walls (array) + total forces for each wall (array) + particles currently colliding with a wall (array)
@njit(fastmath = True, parallel = True)
def calculate_forces(N, radius, spring, g, cell_length, length, X, Y, cell_index, box, particle_collision_lists, grid):
    # Forces for all particles
    forces = np.zeros((2, N), dtype = np.float64)
    
    # Force on verticle walls
    v_walls = np.zeros(N, dtype = np.float64)

    # Total forces for each individual wall
    forces_left_wall, forces_right_wall, forces_bottom_wall, forces_upper_wall = 0.0, 0.0, 0.0, 0.0
    
    # List of particles currently colliding with a wall
    wall_collision_particles = np.zeros(N, dtype = np.bool_)
    
    # Condition used to check if two particles have collided
    collision_condition = 4 * radius * radius # = (2 * radius) ** 2

    for particle in prange(N):
        # Position of particle
        particle_X, particle_Y = X[particle], Y[particle]

        # Particle's cell index
        particle_index_X, particle_index_Y = cell_index[0, particle], cell_index[1, particle]

        
        # Wall forces
        # Only calculate collisions for particles less than radius away from a wall
        if particle_X < radius + box[0, 0] or particle_Y < radius + box[0, 1] or box[1, 0] - radius < particle_X or box[1, 1] - radius < particle_Y:
            # Assign wall forces
            f_left = max(0, radius + box[0, 0] - particle_X)
            f_right = max(0, radius + particle_X - box[1, 0])
            f_bottom = max(0, radius + box[0, 1] - particle_Y)
            f_upper = max(0, radius + particle_Y - box[1, 1])
            
            # Adds to forces for each wall
            forces_left_wall += f_left
            forces_right_wall += f_right
            forces_bottom_wall += f_bottom
            forces_upper_wall += f_upper

            # Wall force for particle
            forces[0, particle] += spring * (f_left - f_right)
            forces[1, particle] += spring * (f_bottom - f_upper)

            # Adds verticle wall forces
            v_walls[particle] += f_left - f_right

            # Particle currently colliding with a wall
            wall_collision_particles[particle] = True

    
        # Particle forces
        # Total force of particle
        particle_force_X, particle_force_Y = 0.0, 0.0
        
        # Checks cells around particle's cell
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                neighbour_index = (particle_index_Y + j) * length + (particle_index_X + i) # Note: check and update
                if neighbour_index in grid:
                    # Calculates forces between current particle and all other particles in current cell
                    for neighbour in grid[neighbour_index]:
                        if neighbour != particle:
                            # Position of neighbour
                            neighbour_X, neighbour_Y = X[neighbour], Y[neighbour]
                            
                            # Distance between the two particles (squared)
                            diff_X = particle_X - neighbour_X
                            diff_Y = particle_Y - neighbour_Y
                            distance = diff_X * diff_X + diff_Y * diff_Y
                        
                            # Checks if particles have collided
                            if 0 < distance < collision_condition:
                                # Actual distance
                                distance = math.sqrt(distance)
                
                                # Angle between particles
                                alpha = math.atan2(diff_Y, diff_X)

                                # Forces
                                coeff = spring * (2 * radius - distance)

                                force_X, force_Y = coeff * math.cos(alpha), coeff * math.sin(alpha)
                
                                particle_force_X += force_X
                                particle_force_Y += force_Y

                                # Adding (particle, neighbour) to thread's list
                                if particle < neighbour:
                                    particle_collision_lists[get_thread_id()].append((int64(particle), neighbour))

        # Adds total force of particle to array
        forces[0, particle] += particle_force_X
        forces[1, particle] += particle_force_Y

    
        # Gravity forces
        forces[1, particle] -= g

    
    forces_walls = np.array([forces_left_wall, forces_right_wall, forces_bottom_wall, forces_upper_wall])
    
    return forces, v_walls, forces_walls, wall_collision_particles

# Returns single time step
def SimulationStep(x:np.ndarray, v:np.ndarray, h:float, part:dict, box:np.ndarray, g:float) -> tuple[np.ndarray,np.ndarray]:
    """
    Preforms one single step forward in time

    Args:
        h (float): Time step size
        x (2xN array): Current particle positions (first row are horizontal coordinates, second row are vertical coordinates)
        v (2xN array): Current particle velocities (first row are horizontal coordinates, second row are vertical coordinates)
        part (dictionary['radius','spring']): Stores properties of the particle (radius r of all particles and spring constant of all particles)
        box (2x2 array): Array storing the corners of the box (np.vstack([low,upp]))
        g (float): Single number, the amount g of gravity
        h (float): Lower bound.

    Returns:
        tuple: xnew(2xN array), vnew(2xN array) which are the updated position and velocity
    """
    # N
    N = len(x[0])
    
    # Radius and spring constant
    radius = part["radius"]
    spring = part["spring"]
    
    # x and y coordinates of all particles
    X, Y = x[0], x[1]
    
    # Size of each cell in grid
    cell_length = 2 * radius

    # Number of cells along grid
    length = int(math.ceil((box[1, 1] - box[0, 1]) / cell_length))

    # Key = index of grid (int), Value = indexes of particles (list)
    grid_empty = Dict.empty(key_type = types.int64, value_type = types.ListType(types.int64))
    
    # List of lists for particle collisions, each thread has it's own list (to avoid race condition in force_particle)
    particle_collision_lists = []
    for i in range(get_num_threads()):
        particle_collision_lists.append(List.empty_list(types.UniTuple(types.int64, 2)))

    # Fills grid and assigns each particle to cell
    grid, cell_index = create_grid(N, cell_length, length, X, Y, grid_empty)

    # Forces of all particles
    forces, v_walls, forces_walls, wall_collision_particles = calculate_forces(N, radius, spring, g, cell_length, length, X, Y, cell_index, box, particle_collision_lists, grid)
    
    # Adds all elements in each list in particle_collision_lists to a set
    particle_collision_particles = set().union(*particle_collision_lists)
    
    x_new = x + dt * v + dt * dt * forces # Updated positions
    x_diff = x_new - x # Difference between old and new positions
    v_new = x_diff / dt # Updated velocities

    X_diff, Y_diff = x_diff[0], x_diff[1]
    distance = np.sqrt(X_diff * X_diff + Y_diff * Y_diff) # Distance travelled
    
    return x_new, v_new , forces_walls, distance, wall_collision_particles, particle_collision_particles, v_walls

import numpy as np
import matplotlib.pyplot as plt
import math
import time
from SimulationStep import SimulationStep
from numba import njit

ran = np.random.rand

# Note: Dividing more expensive than multiplying (only matters in large loops) -> get inverse once and multiply with that

# Updates quantity arrays
@njit
def update_quantities(N, n_bins, bin_length_inv, Y, vx, vy, density, temp, v_walls, left_wall, right_wall):
    for particle in range(N):
        # Bin location
        loc = int(Y[particle] * bin_length_inv)

        if n_bins <= loc:
            loc = n_bins - 1
        elif loc < 0:
            loc = 0

        # Updating density
        density[loc] += 1

        
        # Velocity components of particle
        particle_vx, particle_vy = vx[particle], vy[particle]
        
        # Temperature of particle
        particle_temp = (particle_vx * particle_vx + particle_vy * particle_vy) * 0.5

        # Updating temperature
        temp[loc] += particle_temp

        
        # Force for current particle against vertical wall
        v_wall_particle = v_walls[particle]

        # Updating pressure
        if v_wall_particle != 0:
            # v_wall_particle is positive if particle is at left wall and negative if at right wall
            # Left wall
            if v_wall_particle > 0:
                left_wall[loc] += v_wall_particle
            # Right wall
            else:
                right_wall[loc] -= v_wall_particle

# Guess for fitting is y = Ae ^ (-by)
# Returns A and B
def quantity_function(y_mids, data, quantity_name, quantity_func_name):
    # Can't take log of 0
    condition = data > 0
    y = y_mids[condition]
    d = data[condition]

    log_data = np.log(data)
    
    # Use np.polynomial.fit for a linear approx using a log
    # Returns form log(data) = log(A) - B * y
    # https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.fit.html#numpy.polynomial.polynomial.Polynomial.fit
    poly = np.polynomial.polynomial.Polynomial.fit(y, log_data, 1).convert()

    A = np.exp(poly.coef[0])
    B = -poly.coef[1]

    print(f'Estimated {quantity_name}: {quantity_func_name}(y) = {round(A, 4)} * exp(-{round(B, 4)}y)')
    
    return A, B

# Number of Particles
p = 6
N = 4 ** p

# Particle constant radius and elasticity
part = {"radius" : 0.2, "spring" : 250}

# Gravity Constant
g = 0.05

# Timestep
t_end = 40
h = 0.01
loops = int(t_end / h)

# Last number of timesteps to be analysed
nrec = 3500

low = np.array([0, 0])
upp = np.array([10, 10]) * math.sqrt(N)

# Box Dimensions
box = np.vstack([low, upp])

# Initial positions
x = np.vstack([low[0] + ran(1, N) * (upp[0] - low[0]),
               low[1] + ran(1, N) * (upp[1] - low[1])])

# Initial velocities
v_ini = 2.5
v = 2 * (ran(2, N) - 0.5) * v_ini

# Number of bins
n_bins = 25 

# Length of each grid
bin_length = upp[1] / n_bins

# Inverse of bin_length (used in update_quantities loop)
bin_length_inv = 1 / bin_length

# y middle of bins for plot
y_mids = np.arange(bin_length / 2, upp[1], bin_length)

# Density (number of particles) for each bin
density = np.zeros(n_bins)   

# Temperature for each bin
temp = np.zeros(n_bins)

# Pressure on left and right wall
left_wall = np.zeros(n_bins)
right_wall = np.zeros(n_bins)

# Times to measure density, temperature and pressure 
tor_1 = 20
tor_2 = 40

# Used in simulation for times in [tor_1, tor_2]
tor_1_cond, tor_2_cond = int(tor_1 / h), int(tor_2 / h)

# Simulation
for i in range(loops):
    x, v, _, _, _, _, v_walls = SimulationStep(x, v, h, part, box, g)

    if tor_1_cond <= i <= tor_2_cond:
        update_quantities(N, n_bins, bin_length_inv, x[1], v[0], v[1], density, temp, v_walls, left_wall, right_wall)


## Averages of quantities ##
# Average density
density_avg = density / (box[1, 0] * bin_length)

# Average temperatures - https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero
temp_avg = np.divide(temp, density, out = np.zeros_like(temp), where = density != 0)

# Average pressure
tor_diff = tor_2 - tor_1
left_wall_avg = left_wall * h / (tor_diff * bin_length)
right_wall_avg = right_wall * h / (tor_diff * bin_length)

## Fits of quantities ##
# Density fit
density_A, density_B = quantity_function(y_mids, density_avg, 'density', 'd')
density_fit = density_A * np.exp(-density_B * y_mids)

# Temp fit
temp_A, temp_B = quantity_function(y_mids, temp_avg, 'temperature', 'E')
temp_fit = temp_A * np.exp(-temp_B * y_mids)

# Pressure fit
total_wall = left_wall + right_wall
pressure_A, pressure_B = quantity_function(y_mids, total_wall, 'pressure', 'P')
pressure_fit = pressure_A * np.exp(-pressure_B * y_mids)


## Plots of quantities ##
# Density plot
plt.plot(y_mids, density_avg, 'o', label = 'Average denstity at bin')
plt.plot(y_mids, density_fit, label = 'Fitted function')
plt.xlabel('y')
plt.ylabel('Denstity')
plt.legend(loc = 'upper right')
plt.show()

# Temp plot
plt.plot(y_mids, temp_avg, 'o', label = 'Average temperature at bin')
plt.plot(y_mids, temp_fit, label = 'Fitted function')
plt.xlabel('y')
plt.ylabel('Temperature')
plt.legend(loc = 'upper right')
plt.show()

# Pressure plot
# Left wall plot
plt.plot(y_mids, left_wall, 'o', label = 'Pressure on left wall')
plt.xlabel('y')
plt.ylabel('Pressure')
plt.legend(loc = 'upper right')
plt.show()

# Right wall plot
plt.plot(y_mids, right_wall, 'o', label = 'Pressure on right wall')
plt.xlabel('y')
plt.ylabel('Pressure')
plt.legend(loc = 'upper right')
plt.show()

# Both walls plot
plt.plot(y_mids, total_wall, 'o', label = 'Pressure on both walls')
plt.plot(y_mids, pressure_fit, label = 'Fitted function')
plt.xlabel('y')
plt.ylabel('Pressure')
plt.legend(loc = 'upper right')
plt.legend()
plt.show()

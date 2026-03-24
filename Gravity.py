import numpy as np
import matplotlib.pyplot as plt
import math
from SimulationStep import SimulationStep
from numba import njit
from scipy.optimize import curve_fit

np.random.seed(1) # Use consistent seed in plots + results in overleaf for repeatable results
ran = np.random.rand

# Notes:
    # Add coeff printing
    # np.polyfit currently degree of 5, can be lower -> higher = more accuracy, but 5 should probably be max degree. 3 is fine
    # For large t_end's (like 100, where tor1 = 20 and tor2 = 100) Ae^-By might be a better fit for pressure? -> add an additional 'pressure against Ae^-By' plot at the end as well just to test
        # - Use quantity_function from older versions for Ae^-By
        # - If choosing to keep tor1 = 20 and tor2 = 40 then 1 / (a + by) probably better

# Updates quantity arrays
@njit
def update_quantities(N, n_bins, bin_length_inv, Y, vx, vy, density, temp, v_walls, left_wall, right_wall):
    for particle in range(N):
        # Bin location
        loc = math.floor(Y[particle] * bin_length_inv)
        
        # Only look at particles in bins
        if 0 <= loc < n_bins:
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

# Pressure dependence guess
@njit
def pressure_guess(y, a, b):
    return 1 / (a + b * y)

# Number of Particles
p = 8
N = 4 ** p

# Particle constant radius and elasticity
part = {"radius" : 0.2, "spring" : 250}

# Gravity Constant
g = 0.05

# Timestep
t_end = 40
h = 0.01
loops = int(t_end / h)

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
n_bins = 100

# Length of each grid
bin_length = upp[1] / n_bins

# Inverse of bin_length - used in update_quantities loop
bin_length_inv = 1 / bin_length

# y middle of bins for plot
y_mids = np.arange(bin_length / 2, upp[1], bin_length)

# Density for each bin
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

    if tor_1_cond <= i < tor_2_cond:
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
density_coeff = np.polyfit(y_mids, density_avg, 5)
density_fit = np.poly1d(density_coeff)

# Temperature fit
temp_coeff = np.polyfit(y_mids, temp_avg, 5)
temp_fit = np.poly1d(temp_coeff)

# Pressure fit
pressure_left_param, _ = curve_fit(pressure_guess, y_mids, left_wall_avg)
pressure_left_fit = pressure_guess(y_mids, pressure_left_param[0], pressure_left_param[1])

pressure_right_param, _ = curve_fit(pressure_guess, y_mids, right_wall_avg)
pressure_right_fit = pressure_guess(y_mids, pressure_right_param[0], pressure_right_param[1])


## Plots of quantities ##
# Density plot
plt.plot(y_mids, density_avg, 'o', label = 'Average denstity at bin')
plt.plot(y_mids, density_fit(y_mids), label = 'Fitted function')
plt.xlabel('y')
plt.ylabel('Denstity')
plt.legend(loc = 'upper right')
plt.show()

# Temperature plot
plt.plot(y_mids, temp_avg, 'o', label = 'Average temperature at bin')
plt.plot(y_mids, temp_fit(y_mids), label = 'Fitted function')
plt.xlabel('y')
plt.ylabel('Temperature')
plt.legend(loc = 'upper right')
plt.show()

# Pressure plots
# Left wall plot
plt.plot(y_mids, left_wall_avg, 'o', label = 'Pressure on left wall')
plt.plot(y_mids, pressure_left_fit, label = 'Fitted function')
plt.xlabel('y')
plt.ylabel('Pressure')
plt.legend(loc = 'upper right')
plt.show()

# Right wall plot
plt.plot(y_mids, right_wall_avg, 'o', label = 'Pressure on right wall')
plt.plot(y_mids, pressure_right_fit, label = 'Fitted function')
plt.xlabel('y')
plt.ylabel('Pressure')
plt.legend(loc = 'upper right')
plt.show()

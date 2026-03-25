import numpy as np
import matplotlib.pyplot as plt
import math
from SimulationStep import SimulationStep
from numba import njit
from scipy.optimize import curve_fit

#np.random.seed(1)
ran = np.random.rand

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

# Guess - f(y) = 1 / (a + by)
@njit
def quantity_guess_frac(y, a, b):
    return 1 / (a + b * y)

# Guess - f(y) = Ae ^ (-By)
def quantity_guess_exp(y_mids, data):
    # Can't take log of 0
    condition = data > 0
    y = y_mids[condition]
    d = data[condition]

    log_data = np.log(d)

    # Use np.polynomial.fit for a linear approx using a log
    # Returns form log(data) = log(A) - B * y
    # https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.fit.html#numpy.polynomial.polynomial.Polynomial.fit
    poly = np.polynomial.polynomial.Polynomial.fit(y, log_data, 1).convert()

    A = np.exp(poly.coef[0])
    B = -poly.coef[1]

    return A, B

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
# d(y) = 5th deg polynomial
density_poly_coeff = np.polyfit(y_mids, density_avg, 5)
density_poly_fit = np.poly1d(density_poly_coeff)
print(f'd(y) = polynomial, coefficients: {density_poly_coeff}')

# d(y) = Ae ^ (-By)
density_exp_A, density_exp_B = quantity_guess_exp(y_mids, density_avg)
density_exp_fit = density_exp_A * np.exp(-density_exp_B * y_mids)
print(f'For d(y) = Ae ^ (-By) on left wall, A = {density_exp_A} and B = {density_exp_B}\n')

# Temperature fit
# E(y) = 5th deg polynomial
temp_poly_coeff = np.polyfit(y_mids, temp_avg, 5)
temp_poly_fit = np.poly1d(temp_poly_coeff)
print(f'E(y) coefficients: {temp_poly_coeff}')

# d(y) = Ae ^ (-By)
temp_exp_A, temp_exp_B = quantity_guess_exp(y_mids, temp_avg)
temp_exp_fit = temp_exp_A * np.exp(-temp_exp_B * y_mids)
print(f'For E(y) = Ae ^ (-By) on left wall, A = {temp_exp_A} and B = {temp_exp_B}\n\n')

# Pressure fit
# P(y) = 1 / (a + by)
pressure_left_param_frac, _ = curve_fit(quantity_guess_frac, y_mids, left_wall_avg)
pressure_left_fit_frac = quantity_guess_frac(y_mids, pressure_left_param_frac[0], pressure_left_param_frac[1])
print(f'For P(y) = 1 / (a + by) on left wall, a = {pressure_left_param_frac[0]} and b = {pressure_left_param_frac[1]}')

pressure_right_param_frac, _ = curve_fit(quantity_guess_frac, y_mids, right_wall_avg)
pressure_right_fit_frac = quantity_guess_frac(y_mids, pressure_right_param_frac[0], pressure_right_param_frac[1])
print(f'For P(y) = 1 / (a + by) on right wall, a = {pressure_right_param_frac[0]} and b = {pressure_right_param_frac[1]}\n')

# P(y) = Ae ^ (-By)
pressure_left_param_exp_A, pressure_left_param_exp_B = quantity_guess_exp(y_mids, left_wall_avg)
pressure_left_fit_exp = pressure_left_param_exp_A * np.exp(-pressure_left_param_exp_B * y_mids)
print(f'For P(y) = Ae ^ (-By) on left wall, A = {pressure_left_param_exp_A} and B = {pressure_left_param_exp_B}')

pressure_right_param_exp_A, pressure_right_param_exp_B = quantity_guess_exp(y_mids, right_wall_avg)
pressure_right_fit_exp = pressure_right_param_exp_A * np.exp(-pressure_right_param_exp_B * y_mids)
print(f'For P(y) = Ae ^ (-By) on right wall, A = {pressure_right_param_exp_A} and B = {pressure_right_param_exp_B}')

## Plots of quantities ##
# Density plot
plt.plot(y_mids, density_avg, 'o', label = 'Average denstity at bin')
plt.plot(y_mids, density_poly_fit(y_mids), label = 'Fitted function, d(y) = 5th deg polynomial')
plt.plot(y_mids, density_exp_fit, label = 'Fitted function, d(y) = Ae ^ (-By)')
plt.title(f"Density against y", fontsize = 30)
plt.xlabel('y', fontsize = 25)
plt.ylabel('Denstity', fontsize = 25)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.legend(loc = 'upper right', fontsize = 25)
plt.show()

# Temperature plot
plt.plot(y_mids, temp_avg, 'o', label = 'Average temperature at bin')
plt.plot(y_mids, temp_poly_fit(y_mids), label = 'Fitted function, d(y) = 5th deg polynomial')
plt.plot(y_mids, temp_exp_fit, label = 'Fitted function, E(y) = Ae ^ (-By)')
plt.title(f"Temperature against y", fontsize = 30)
plt.xlabel('y', fontsize = 25)
plt.ylabel('Temperature', fontsize = 25)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.legend(loc = 'lower left', fontsize = 25)
plt.show()

# Pressure plots
# Left wall plot
plt.plot(y_mids, left_wall_avg, 'o', label = 'Average left wall pressure at bin')
plt.plot(y_mids, pressure_left_fit_frac, label = 'Fitted function, P(y) = 1 / (a + by)')
plt.plot(y_mids, pressure_left_fit_exp, label = 'Fitted function, P(y) = Ae ^ (-By)')
plt.title(f"Pressure of left wall against y", fontsize = 30)
plt.xlabel('y', fontsize = 25)
plt.ylabel('Pressure', fontsize = 25)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.legend(loc = 'upper right', fontsize = 25)
plt.show()

# Right wall plot
plt.plot(y_mids, right_wall_avg, 'o', label = 'Average right wall pressure at bin')
plt.plot(y_mids, pressure_right_fit_frac, label = 'Fitted function, P(y) = 1 / (a + by)')
plt.plot(y_mids, pressure_right_fit_exp, label = 'Fitted function, P(y) = Ae ^ (-By)')
plt.title(f"Pressure of right wall against y", fontsize = 30)
plt.xlabel('y', fontsize = 25)
plt.ylabel('Pressure', fontsize = 25)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.legend(loc = 'upper right', fontsize = 25)
plt.show()

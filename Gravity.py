
import numpy as np
from math import *
rand = np.random.rand
import matplotlib.pyplot as plt
import SimulationStep as sim
from numba import njit, prange, get_num_threads, get_thread_id

# Set up values 


# Number of Particles
p=8
N = 4**p

# Maximum p value to be run for part b u
pmax = p

# Particle Constant Radius and Elasticity
part = {
    "radius" : 0.2,
    "spring" : 250
}

# Timestep
tini = 0
tend=40
h = 0.01
loops = int(tend/h)
# Last number of timesteps to be analysed
nrec = 3500

t=np.arange(tini,tend,h)

low = np.array([0,0])

upp = np.array([10,10]) * np.sqrt(N)

# Box Dimensions
box = np.vstack([low,upp])

# Gravity Constant
g = 0.05

# x
x=np.vstack([low[0]+rand(1,N)*(upp[0]-low[0]),
low[1]+rand(1,N)*(upp[1]-low[1])])

# Inital Velocity
vini = 2.5

# v
v=2*(rand(2,N)-0.5)*vini

#fig, ax = plt.subplots()
#scatter  = ax.scatter(x[0],x[1],s=5) 

#ax.set_xlim(0,upp[0])
#ax.set_ylim(0,upp[0])


# code for density 
# split y axis in to 25 lots can change
nbins = 25 
density_total = np.zeros(nbins)   
temp_total = np.zeros(nbins)   
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
# actually set up bin assigning
y_bins = np.linspace(low[1],upp[1],nbins+1)
# for plotting density you want the centre of the box for y co-ord
y_middles = 0.5 *(y_bins[:-1]+y_bins[1:])
def density_calc(x,low,upp,y_bins):

    
    # count particles in bin , _ is for unused output of np.histogram
    count, _ = np.histogram(x[1,:],bins=y_bins)
    # work out area of bins for density calc
    lot_h = y_bins[1] - y_bins[0]
    width = upp[0] -low[0]
    area = width*lot_h
    # density calc
    density = count/area
    return density
# code for function fitting # i think this is correct not sure 
# guess for fitting is y = A.e^-By so need logs 
# expect lots of particles at bottom and few at top
def density_function(y_middles,density_avg):
    
    # remove any 0 as will be using logs due to guess of Exp function
    data = density_avg > 0 
    y = y_middles[data]
    d = density_avg[data]
    # log of density 
    logd = np.log(d)
    # use np.polynomial.fit for a linear approx using a log 
    # gives in form log(density) = log(A) - B*y 
    # Source https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.fit.html#numpy.polynomial.polynomial.Polynomial.fit
    poly = np.polynomial.polynomial.Polynomial.fit(y,logd,1)
    poly = poly.convert()
    
   
    A = np.exp(poly.coef[0])
    B = -poly.coef[1]
    print(f'Estimated denstity function: d(y)= {A:.4f} * exp(-{B:.4f}y)')
    # return A, B for plotting again actual denstity 
    return A,B



def temp_calc(x,v,y_bins):
    # particle y pos
    y = x[1,:]
    # k.e par part
    ke = 0.5 * (v[0,:]*v[0,:] + v[1,:]*v[1,:])
    # count particles in bins , _ is for unused output of np.histogram
    count, _ = np.histogram(x[1,:],bins=y_bins)
    # sum of ke in each bin 
    energy_count, _ =  np.histogram(y, bins=y_bins, weights=ke)
    temp = np.zeros_like(energy_count)
    
    # remove all bins with zero particles
    temp [count > 0] = energy_count[count > 0] / count[count > 0]
    return temp
# this is just a straight line no dependence which i think makes sense as temp is averaged and doesnt rely on number of particles like denstity and pressure do just speed and they 
# are all effected by gravity the same so will just fit a straight line to this
def temp_fitting(temp):
    data = temp > 0 
    
    t = temp[data]
    T = np.mean(t)
    print(f"Estimated temperature: T ≈ {T:.4f}")
    return T


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

# there really is not effect that gravity has because it is far too low and our box is far to big

def pressure_function(ymids, leftWall, rightWall):

    pressure = leftWall + rightWall

    # remove zeros
    data = pressure > 0
    y = ymids[data]
    p = pressure[data]

    logp = np.log(p)

    # fit log P = log C - b y
    poly = np.polynomial.polynomial.Polynomial.fit(y, logp, 1)
    poly = poly.convert()

    C = np.exp(poly.coef[0])
    b = -poly.coef[1]

    print(f'Estimated pressure: P(y)= {C:.4e} * exp(-{b:.4e} y)')

    return C, b
for i in range(loops):
    x, v, _, _, _, _, vWalls = sim.SimulationStep(x, v, h, part, box, g)

    # after eq
    if i> nrec:
        density = density_calc(x,low,upp,y_bins)
        density_total += density
        calculate_pressure(N, nbins, binLength, x[1], vWalls, leftWall, rightWall)
        temp = temp_calc(x,v,y_bins)
        temp_total += temp

        #data = np.c_[np.array(x[0]),np.array(x[1])]
        #scatter.set_offsets(data)
        #plt.pause(0.01)





# work out average density in your lots across time 
density_avg = density_total/(loops-nrec)

A,B = density_function(y_middles,density_avg)
#C,D = density_function(y_middles,density_avg)

fit = A*np.exp(-B * y_middles)

plt.plot(y_middles,density_avg,'o',label = 'simulated denstity')
plt.plot(y_middles,fit,label = 'fitted function')
plt.xlabel('height y')
plt.ylabel('denstity')
plt.legend()
plt.show()
# temp plotting

T = temp_fitting(temp)
temp_avg = temp_total /(loops - nrec)
plt.plot(y_middles,temp_avg,'o')
plt.axhline(T,label ='constant fit')
plt.show()






#Time pressure was taken accross
torDiff = loops-nrec


# Average pressure across time 
leftWall *= h / torDiff
rightWall *= h / torDiff
C,D = pressure_function(yMids,leftWall,rightWall)
fit2 = C * np.exp(-D * yMids)
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

plt.xlabel("Vertical height y")
plt.ylabel("Pressure")
plt.plot(yMids,fit2,label = 'fitted function')
plt.plot(yMids,leftWall+rightWall,'o', label ='pressure across both walls against vertical height')
plt.legend()
plt.show()
print("Mean T =", np.mean(temp_avg))
print("b (density) =", B)
print("g/T =", g / np.mean(temp_avg))
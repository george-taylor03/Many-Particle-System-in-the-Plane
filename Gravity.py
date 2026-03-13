
import numpy as np
from math import *
rand = np.random.rand
import matplotlib.pyplot as plt
import SimulationStep as sim
import time
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
# actually set up lot assigning
y_bins = np.linspace(low[1],upp[1],nbins+1)
# for plotting density you want the centre of the box for y co-ord
y_middles = 0.5 *(y_bins[:-1]+y_bins[1:])
def density_calc(x,low,upp,y_bins):

    
    # count particles in lots , _ is for unused output of np.histogram
    count, _ = np.histogram(x[1,:],bins=y_bins)
    # work out area of lots for density calc
    lot_h = y_bins[1] - y_bins[0]
    width = upp[0] -low[0]
    area = width*lot_h
    # density calc
    density = count/area
    return density
# code for function fitting # i think this is correct not sure 
# guess for fitting is y = A.e^-By so need logs
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

for i in range(loops):
    x , v = sim.SimulationStep(x, v, h, part, box, g)[0:2]

    # after eq
    if i> nrec:
        density = density_calc(x,low,upp,y_bins)
        density_total += density
        
    #if i%5==0:
        #data = np.c_[np.array(x[0]),np.array(x[1])]
        #scatter.set_offsets(data)
        #plt.pause(0.01)
#plt.show()


# work out average density in your lots across time 
density_avg = density_total/(loops-nrec)
A,B = density_function(y_middles,density_avg)
# plot of denstity again estimated denstity.
fit = A*np.exp(-B * y_middles)

plt.plot(y_middles,density_avg,'o',label = 'simulated denstity')
plt.plot(y_middles,fit,label = 'fitted function')
plt.xlabel('height y')
plt.ylabel('denstity')
plt.legend()
plt.show()
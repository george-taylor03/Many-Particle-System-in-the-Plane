# imports

import numpy as np
from math import *
rand = np.random.rand
import matplotlib.pyplot as plt
import SimulationStep as sim
import time
# Set up values 


# set up conditions to run the different experiments for T2
# change to true or false depending on which you want to run (to stop unneeded waiting )
# can all be true at once 
# for part a 
parta = True
# for part 
partb = True
# for part c 
partc = True 



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
tend=20
h = 0.01
loops = int(tend/h)
# Last number of timesteps to be analysed
nrec = 1000

t=np.arange(tini,tend,h)

low = np.array([0,0])

upp = np.array([10,10]) * np.sqrt(N)

# Box Dimensions
box = np.vstack([low,upp])

# Gravity Constant
g = 0

# x
x=np.vstack([low[0]+rand(1,N)*(upp[0]-low[0]),
low[1]+rand(1,N)*(upp[1]-low[1])])

# Inital Velocity
vini = 2.5

# v
v=2*(rand(2,N)-0.5)*vini

# Define speed function 
def speed(v):
    s = np.sqrt(np.sum(v**2,0))
    return s 
#  Define average temp

def AvgTemp(v):
    return  np.sum(v*v)/(2*v.shape[1])

# Standard deviation of temperature for the last nrec timesteps
def sigma(T):
    return np.std(T[-nrec-1:])








"""" loglog for part a """


if parta == True:
    T = np.zeros((loops))
    S = np.zeros(pmax)
    # Number of updates on time estimation
    #NTimeUpdates = 200

    # Set = 1 for Temperature graphs at each value of p
    TemperatureGraph = 0


    # Constant factor for how much longer the next p will take to compute (5 is roughly correct)
    # Becomes more accurate for p>6 ish
    # Cfactor = 5
    Nplot = []
    # Which timesteps to update time estimation
    #Update = np.arange(1, (loops), int(loops/NTimeUpdates))

    # Collecting data for each p up to pmax
    for p in range(1,pmax+1):

       
        # Number of Particles
        N = 4**p

        # N values for plotting
        Nplot.append(N)
    
        t=np.arange(tini,tend,h)
    
        low = np.array([0,0])
    
        upp = np.array([10,10]) * np.sqrt(N)
    
        #Box Dimensions
        box = np.vstack([low,upp])

        
        # x
        x=np.vstack([low[0]+rand(1,N)*(upp[0]-low[0]),
        low[1]+rand(1,N)*(upp[1]-low[1])])

        # Inital Velocity
        vini = 2.5

        # v
        v=2*(rand(2,N)-0.5)*vini

        # Reseting time estimation using guesses from last p value
        #TimeLeft =  round(Cfactor*TimeTaken,2)
        #CurrentFinishTime = time.strftime("%H:%M:%S", time.localtime(time.time() + TimeLeft))

        # Start timing for p value
        #StartTime = time.perf_counter()
        
        for i in range((loops)):
        
            x , v = sim.SimulationStep(x,v,h,part,box,g)[0:2]
            
            T[i] = AvgTemp(v)

            # Time estimation
            #if i in Update:
            #    CurrentTime = time.perf_counter()
            #    TimeLeft = (CurrentTime - StartTime) * ((loops) - 1 - i) / i
            #    CurrentFinishTime = time.strftime("%H:%M:%S", time.localtime(time.time() + TimeLeft))
            #    print(f"\r Time Left: {round(TimeLeft,2)}, finishing at: {CurrentFinishTime}", end="", flush=True)
            
        print()
        
        S[p-1] = sigma(T)

        # Finishing timing
        #EndTime = time.perf_counter()
        #TimeTaken = EndTime - StartTime
        # Time to finish all p values to pmax
        #TotalTime = TimeTaken * sum(Cfactor**n for n in range(1, pmax-p+1))
        #FinishTime = time.strftime("%H:%M:%S", time.localtime(time.time() + TotalTime))
        
        #print(f"p / N = {p} / {4**p} out of pmax = {pmax}. Time taken to compute = {round(TimeTaken,2)}. Aprrox next time to compute = {round(Cfactor*TimeTaken,2)} finishing at {FinishTime}")
        print(f"Over the last {nrec} timesteps, average temperature = {T[-nrec-1:].mean()} and range as a fraction of the average = {(T[-nrec-1:].max() - T[-nrec-1:].min()) / T[-nrec-1:].mean()}")
        
        # Temperature graphs for each p value
        if TemperatureGraph == 1:
            plt.plot(np.arange(0,loops,1), T, label = 4**p)
            plt.xlabel('Timestep')
            plt.ylabel('Temperature')
            plt.legend()
            plt.show()

    # Plotting N versus standard deviation on a loglog scale
    plt.loglog(Nplot,S,'o-' )
    plt.xlabel('Number of particles')
    plt.ylabel('Standard deviation')
    plt.show()


""" histogram for part b """
if partb == True:
    for i in range(loops):
        x , v = sim.SimulationStep(x, v, h, part, box, g)[0:2]

    s = speed(v)
    # number of bins not sure what final value will be 
    bins = 30
    plt.hist(s,bins=bins,density=True)
    plt.ylabel('number of particles')
    plt.xlabel('final speed value')
    plt.show()

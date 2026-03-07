#Imports 
import numpy as np
import matplotlib.pyplot as plt
import SimulationStep as sim
import time
from math import *
from scipy.constants import Boltzmann
rand = np.random.rand

###############################################
#Things that can be played around with:

#Maximum p value to be run until
pmax = 7

#Last number of timesteps to be analysed
nrec = 1500

#Number of updates on time estimation
NTimeUpdates = 200

#Set = 1 for Temperature graphs at each value of p
TemperatureGraph = 0


#Constant factor for how much longer the next p will take to compute (5 is roughly correct)
#Becomes more accurate for p>6 ish
Cfactor = 5

#Timestep start, end and size
tini = 0
tend=20
dt = 0.01

###############################################

#Number of timesteps
ndt = int(tend/dt)

#Which timesteps to update time estimation
Update = np.arange(1, ndt, int(ndt/NTimeUpdates))

#Average temperature of every particle
def AvgTemp(v):
    return 2 * np.average((v[0]**2 + v[1]**2) / 2) / (3 * Boltzmann)

#Standard deviation of temperature for the last nrec timesteps
def sigma(T):
    return np.std(T[-nrec-1:])

#Particle Constant Radius and Elasticity
part = {
    "radius" : 0.2,
    "spring" : 250
}

#Setting variables
TimeLeft = -1
CurrentFinishTime = -1
TimeTaken = -1

T = np.zeros(ndt)
S = np.zeros(pmax)

Nplot = []

#Collecting data for each p up to pmax
for p in range(1,pmax+1):

    #Number of Particles
    N = 4**p

    #N values for plotting
    Nplot.append(N)
    
    t=np.arange(tini,tend,dt)
    
    low = np.array([0,0])
    
    upp = np.array([10,10]) * np.sqrt(N)
    
    #Box Dimensions
    box = np.vstack([low,upp])
    
    #Gravity Constant
    g = 0

    #x
    x=np.vstack([low[0]+rand(1,N)*(upp[0]-low[0]),
    low[1]+rand(1,N)*(upp[1]-low[1])])

    #Inital Velocity
    vini = 250

    # v
    v=2*(rand(2,N)-0.5)*vini

    #Reseting time estimation using guesses from last p value
    TimeLeft =  round(Cfactor*TimeTaken,2)
    CurrentFinishTime = time.strftime("%H:%M:%S", time.localtime(time.time() + TimeLeft))

    #Start timing for p value
    StartTime = time.perf_counter()
    
    for i in range(ndt):
    
        x , v = sim.SimulationStep(x,v,dt,part,box,g)
        
        T[i] = AvgTemp(v)

        #Time estimation
        if i in Update:
            CurrentTime = time.perf_counter()
            TimeLeft = (CurrentTime - StartTime) * (ndt - 1 - i) / i
            CurrentFinishTime = time.strftime("%H:%M:%S", time.localtime(time.time() + TimeLeft))
            print(f"\r Time Left: {round(TimeLeft,2)}, finishing at: {CurrentFinishTime}", end="", flush=True)
        
    print()
    
    S[p-1] = sigma(T)

    #Finishing timing
    EndTime = time.perf_counter()
    TimeTaken = EndTime - StartTime
    #Time to finish all p values to pmax
    TotalTime = TimeTaken * sum(Cfactor**n for n in range(1, pmax-p+1))
    FinishTime = time.strftime("%H:%M:%S", time.localtime(time.time() + TotalTime))
    
    print(f"p / N = {p} / {4**p} out of pmax = {pmax}. Time taken to compute = {round(TimeTaken,2)}. Aprrox next time to compute = {round(Cfactor*TimeTaken,2)} finishing at {FinishTime}")
    print(f"Over the last nrec timesteps, average temperature = {T[-nrec-1:].mean()} and range as a fraction of the average = {(T[-nrec-1:].max() - T[-nrec-1:].min()) / T[-nrec-1:].mean()}")
    
    #Temperature graphs for each p value
    if TemperatureGraph == 1:
        plt.plot(np.arange(0,ndt,1), T, label = 4**p)
        plt.xlabel('Timestep')
        plt.ylabel('Temperature')
        plt.legend()
        plt.show()

#Plotting N versus standard deviation on a loglog scale
plt.loglog(S, Nplot)
plt.xlabel('Standard deviation')
plt.ylabel('Number of particles')
plt.show()









#Imports 
import numpy as np
import matplotlib.pyplot as plt
import SimulationStep as sim
import time
from math import *
rand = np.random.rand

###############################################
#Things that can be played around with:

#p value, number of particles N = 4^p
p = 8

#Last number of timesteps to be analysed
nrec = 500

#Number of updates on time estimation
NTimeUpdates = 200

#Timestep start, end and size
tini = 0
tend=40
dt = 0.01

#Gravity
g = 0.05

#Number of bins along the y axis
Nbins = 25

###############################################

#Average temperature of given particles
def AvgTemp(v):
    return np.mean(v[0]**2 + v[1]**2) / 2

#Number of timesteps
ndt = int(tend/dt)

#Which timesteps to update time estimation
Update = np.arange(1, ndt, int(ndt/NTimeUpdates))

#Number of Particles
N = 4**p

BinLength = 10*sqrt(N) / Nbins

BinBoundary = np.linspace(10*sqrt(N), 0, Nbins)

#2D array for each timestep and each bin
Temperature = np.zeros((ndt,Nbins))

#1D array for average temperature in each bin
FinalTemperature = np.zeros(Nbins)

t = np.arange(tini,tend,dt)
    
low = np.array([0,0])
    
upp = np.array([10,10]) * np.sqrt(N)
    
#Box Dimensions
box = np.vstack([low,upp])

#Particle Constant Radius and Elasticity
part = {
    "radius" : 0.2,
    "spring" : 250
}
  
#x
x=np.vstack([low[0]+rand(1,N)*(upp[0]-low[0]),
low[1]+rand(1,N)*(upp[1]-low[1])])

#Inital Velocity
vini = 250

# v
v=2*(rand(2,N)-0.5)*vini

#Start timing for p value
StartTime = time.perf_counter()




for i in range(ndt):

    x, v, *_ = sim.SimulationStep(x,v,dt,part,box,g)

    for j in range(Nbins):    
        indexs = np.where(np.digitize(x[1],BinBoundary) - 1 == j)
        if np.size(indexs) == 0:
            Temperature[i][j] = 0
            print('bin with no particles')
        else:
            Temperature[i][j] = AvgTemp(np.vstack((v[0][indexs],v[1][indexs])))




    #Time estimation
    if i in Update:
        CurrentTime = time.perf_counter()
        TimeLeft = (CurrentTime - StartTime) * (ndt - 1 - i) / i
        CurrentFinishTime = time.strftime("%H:%M:%S", time.localtime(time.time() + TimeLeft))
        print(f"\r Time Left: {round(TimeLeft,2)}, finishing at: {CurrentFinishTime}", end="", flush=True)
    
print()

for j in range(Nbins):
    FinalTemperature[j] = np.mean(Temperature[j])

plt.plot(FinalTemperature, np.arange(0,Nbins,1))
plt.show()







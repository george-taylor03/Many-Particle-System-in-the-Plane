#Imports 

import numpy as np
from math import *
# np.random.seed(1)
rand = np.random.rand
import matplotlib.pyplot as plt
import SimulationStep as sim
from matplotlib.patches import Rectangle


#Number of Particles
p=3
N = 4**p

#Particle Constant Radius and Elasticity
part = {
    "radius" : 0.2,
    "spring" : 250
}


#Timestep
tini = 0
tend=100   
h = 0.01
loops = tend/h

t=np.arange(tini,tend,h)

#Set up box dimensions
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
vini = 2.5

# v
v=2*(rand(2,N)-0.5)*vini

# #Set up graph
fig, ax = plt.subplots()
scatter  = ax.scatter(x[0],x[1],s=2) 

#Create moving line
boxline = ax.plot([0,upp[0]],[upp[1],upp[1]])[0]

# boxline = ax.plot([0,upp[0]],[low[1],low[1]])[0]


ax.add_patch(Rectangle(low, upp[0] - low[0], upp[1] - low[1], linewidth = 1, fill = False))

diff = [(upp[0] - low[0]) * 0.1, (upp[1] - low[1]) * 0.1]
plt.xlim(low[0] - diff[0], upp[0] + diff[0])
plt.ylim(low[1] - diff[1], upp[1] + diff[1])

#Wall speed
a=50
#Settled Time tor
tor1 = 5
tor2 = tor1 + (5 * np.sqrt(N))/a

#Two time periods after settled from compression
tor3 = 50
tor4 = tor3 + 20

torTimes = np.arange(tor1,tor2,h)

#Recorded times of temperature
tempTimes = np.arange(tor3,tor4,h)

#Recorded times of pressure
pressTimes = []

#Array of temperatures during tor3 and tor4
T = []
#

#Array of pressure per wall initially before compression
pInit = np.zeros(4)

#Array of pressure per wall when compressed
pComp = np.zeros(4)

#Box Pressure values
pBox=[]

#Vary different wall speeds
for i in range(int(loops)):
    #Change box dimensions
    if i >= int(tor1/h) and i<int(tor2/h):

        #Locates index of tor times. Mods i so it starts from 0 and divides tor1 so it is inline with loops
        loc = i-int(tor1/h)

        upp[1] = 10 * np.sqrt(N) - a*(torTimes[loc] - tor1)



    #Box Dimensions
    box = np.vstack([low,upp])

    #Run Simulation
    x , v, forceW,_,_,_,_ = sim.SimulationStep(x, v, h, part, box, g)

        #Records initial pressure
    if i<int(tor1/h):
        #Pressure at each wall
        pInit[0] += forceW[0] / box[1,1]
        pInit[1] += forceW[1] / box[1,1]
        pInit[2] += forceW[2] / box[1,0]
        pInit[3] += forceW[3] / box[1,0]
    
    #Measure Temperature and pressure after at tor3 and tor4
    elif i >= int(tor3/h) and i<int(tor4/h):
        T.append(np.average((v[0]**2 + v[1]**2)/2))

        # #Pressure at each wall
        pComp[0] += forceW[0] / box[1,1]
        pComp[1] += forceW[1] / box[1,1]
        pComp[2] += forceW[2] / box[1,0]
        pComp[3] += forceW[3] / box[1,0]

        #Finds pressure over serverl timesteps
        if (i-int(tor3/h)) % 20 == 0:
            #Average perssure over serveral timesteps
            pComp = pComp / 20
            #Total box pressure
            pBox.append(np.average(pComp))

            #Reset Pressure
            pComp = np.zeros(4)

            #Add recorded pressure time. Muktiple by h to get to normal time not per timestep
            pressTimes.append(i*h)

    if i%1==0:
        boxline.set_ydata([upp[1],upp[1]])

        # boxline.set_ydata([low[1],low[1]])
        data = np.c_[np.array(x[0]),np.array(x[1])]
        scatter.set_offsets(data)
        plt.pause(0.00001)

#Initial Pressure. To get average must divide the number of timesteps it was taken over
print(f"Inital Average Pressure {np.average(pInit / (tor1/h))}")

#Compressed average Pressure
print(f"Compressed Average Pressure {np.average(pBox)}")

#Average Temp over tor3 and tor4
print(f"Average Compressed Temperature {np.average(T)}")

#Plot temperature of box between tor3 and tor 4
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.plot(tempTimes,T,label = f"Temperature over Time for Wall Speed a = {a}")
plt.legend()
plt.show()

#Plot plot average pressure of all wals between tor3 and tor4
plt.xlabel("Time")
plt.ylabel("Pressure")
plt.plot(pressTimes,pBox, label = f"Pressure over Time for Wall Speed a = {a}")
plt.legend()
plt.show()
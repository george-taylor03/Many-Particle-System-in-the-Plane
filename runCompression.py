import numpy as np
import matplotlib.pyplot as plt
import math
from SimulationStep import SimulationStep
from matplotlib.patches import Rectangle

ran = np.random.rand    
    
# Number of Particles
p = 7
N = 4 ** p

sqrt_N = math.sqrt(N)

# Particle constant radius and elasticity
part = {"radius" : 0.2, "spring" : 250}

# Gravity Constant
g = 0

# Timestep
t_end = 100
h = 0.01
loops = int(t_end / h)

low = np.array([0, 0])
upp = np.array([10, 10]) * sqrt_N

# Box Dimensions
box = np.vstack([low, upp])

# Initial positions
x = np.vstack([low[0] + ran(1, N) * (upp[0] - low[0]),
               low[1] + ran(1, N) * (upp[1] - low[1])])

# Initial velocities
v_ini = 2.5
v = 2 * (ran(2, N) - 0.5) * v_ini

# Wall speed
a = 50

# Settled Time tor
tor1 = 5
tor2 = tor1 + (5 * sqrt_N) / a

tor1_loopNum, tor2_loopNum = int(tor1 / h), int(tor2 / h)

# Two time periods after settled from compression
tor3 = 50
tor4 = tor3 + 20

tor3_loopNum, tor4_loopNum = int(tor3 / h), int(tor4 / h)

torTimes = np.arange(tor1, tor2, h)

# Recorded times of temperature
tempTimes = np.arange(tor3, tor4, h)

# list of temperatures during tor3 and tor4
T = []

# Recorded times of pressure
pressTimes = []

# Array of pressure per wall initially before compression
pInit = np.zeros(4)

# Array of pressure per wall when compressed
pComp = np.zeros(4)

# Box Pressure values
pBox = []

# Set up graph
fig, ax = plt.subplots()
scatter = ax.scatter(x[0], x[1], s = 2) 

# Box for plotting
box_border = ax.add_patch(Rectangle(low, upp[0] - low[0], upp[1] - low[1], linewidth = 1, fill = False))

diff = [(upp[0] - low[0]) * 0.1, (upp[1] - low[1]) * 0.1]
plt.xlim(low[0] - diff[0], upp[0] + diff[0])
plt.ylim(low[1] - diff[1], upp[1] + diff[1])

# Simulation
for i in range(loops):
    # Changing box dimensions
    if tor1_loopNum <= i < tor2_loopNum:
        # Index of tor times
        loc = i - tor1_loopNum

        # Updating box
        box[1, 1] = 10 * sqrt_N - a * (torTimes[loc] - tor1)
    
    # Run Simulation
    x, v, forceW, *_ = SimulationStep(x, v, h, part, box, g)
    
    # Records initial pressure (for t in [tor1, tor2])
    if i < tor1_loopNum:
        # Length of each wall
        box_length = np.array([box[1, 1], box[1, 1], box[1, 0], box[1, 0]])
        
        # Pressure for each wall
        pInit += np.divide(forceW, box_length)
    
    # Measure average temperature and pressure (for t in [tor3, tor4])
    elif tor3_loopNum <= i < tor4_loopNum:
        # Average temperature
        vx, vy = v[0], v[1]
        T.append(np.average((vx * vx + vy * vy) / 2))

        # Length of each wall
        box_length = np.array([box[1, 1], box[1, 1], box[1, 0], box[1, 0]])
        
        # Pressure for each wall
        pComp += np.divide(forceW, box_length)
        
        # Finds average pressure over several timesteps
        if (i - tor3_loopNum) % 20 == 0:
            # Average pressure over several timesteps
            pComp /= 20
            
            # Total box pressure
            pBox.append(np.average(pComp))

            # Reset pressure
            pComp = np.zeros(4)

            # Add recorded pressure time
            pressTimes.append(i * h)
    
    box_border.set_height(box[1, 1])
    scatter.set_offsets(np.c_[x[0], x[1]])
    plt.pause(0.01)

plt.show()

# Initial Pressure
print(f"Initial Average Pressure {np.average(pInit / tor1_loopNum)}")

# Compressed average pressure
print(f"Compressed Average Pressure {np.average(pBox)}")

# Average temperature over tor3 and tor4
print(f"Average Compressed Temperature {np.average(T)}")

# Temperature plot of box between tor3 and tor4
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.plot(tempTimes, T, label = f"Temperature over time for wall speed a = {a}")
plt.legend()
plt.show()

# Average pressure plot of all walls between tor3 and tor4
plt.xlabel("Time")
plt.ylabel("Pressure")
plt.plot(pressTimes, pBox, label = f"Pressure over time for wall apeed a = {a}")
plt.legend()
plt.show()

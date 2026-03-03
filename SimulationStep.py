#Imports 

import numpy as np
from math import *


def gridBuilder(x,box,N):
    pass
    #Assign gridLength
    gridLength = 0.4

    #Get number of grid rows and columns
    rows = int(box[1][0] / gridLength)
    cols = int(box[1][0] / gridLength)

    #Zero matrix representing grid positions
    gridArray = []

    # build more effective grid assignment build on gridarrys assignments using python dictionaries 
    grid = {} # empty dict to be appended

    #Assign Grids
    for particle in range(N):
        #Determine Grid Row

        gridPosRow = int(x[0][particle] // gridLength)

        #Accounting for boundary cases
        if gridPosRow>(rows-1):
            gridPosRow -= 1

        gridPosCol = int(x[1][particle] // gridLength)

        #Accounting for boundary cases
        if gridPosCol>(cols-1):
            gridPosCol -= 1

        #Append gridPosition to particles index
        gridArray.append([gridPosRow,gridPosCol])

        # if a box exists just add the particle to the box
        if (gridPosRow,gridPosCol) in grid:
            grid[gridPosRow,gridPosCol].append(particle)
        # if not add a new box 
        else:
            grid[gridPosRow,gridPosCol] = [particle]
    # this sorts much faster as it uses the very efficient dictionaries lookups

    return grid,gridArray,gridLength
        
# check each particle against once in its box  or the surrounding 8 boxes 
def connect(grid, gridArray,N):
    # create a empty list for each particle to store its nearby particles 
    # just a list of list [[],[],[],[],[]] <- like this
    partnet = [[] for j in range(N)]
    # split our grid array values to into two separate pairs instead of tuples for our dict
    for i in range(N):
        xco = gridArray[i][0]
        yco = gridArray[i][1]
        # checks the surrounding 3X3 boxes around this particles 
        # better that nested if loops as for loops are faster 
        for dxco in (-1,0,1):
            for dyco in (-1,0,1):
                # work out the nearby box co-ords e.g if we in box (4,4) we check
                # (3,3),(3,4),(3,5),(4,4), ....., (5,5)
                surr = (xco + dxco, yco + dyco)
                # check if there is a particle in the box no point checking empty boxes
                if surr in grid:
                    # add all particles from nearby boxes to out list of nearby particles
                    partnet[i] += grid[surr]
    return partnet 

def forceWall(Forces,x,gridLength,box,part,N):
    #Check collision with wall
    for particle in range(N):
        if x[0][particle] <= gridLength or x[0][particle] >= box[1][0] - gridLength or x[1][particle] <= gridLength or x[1][particle] >= box[1][1] - gridLength:
            #Particle A 
            A = x[:, particle]

            #Wall Forces
            fLeft = max(0,part['radius'] + box[0][0] - A[0])
            fRight = max(0,part['radius'] + A[0] - box[1][0])
            fBot = max(0,part['radius'] + box[0][1] - A[1])
            fUp = max(0,part['radius'] + A[1] - box[1][1])

            #Check Wall Collision
            #X component
            Forces[0][particle] += part['spring'] * (fLeft - fRight)

            #y component
            Forces[1][particle] += part['spring'] * (fBot - fUp)
    return Forces

def forceParticle(Forces,x,partNet,part,N):
    #Cycle through each particle
    for particle in range(N):
        #Cycle through all its connections
        for neighbour in partNet[particle]:
                    #Particle A 
            A = x[:, particle]

            #Other Particle
            B = x[:,neighbour]


            #Distance between them
            d = np.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)

            #Check if will collide
            if d < 2 * part['radius'] and d>0:
                #Angle
                a = np.arctan2(A[1]-B[1],A[0]-B[0])
            
                #X component
                Forces[0][particle] += part['spring'] * (2*part['radius'] - d) * np.cos(a)

                #y component
                Forces[1][particle] += part['spring'] * (2*part['radius'] - d) * np.sin(a)

    return Forces

def SimulationStep(x:np.ndarray, v:np.ndarray, h:float, part:dict, box:np.ndarray, g:float, N:int) -> tuple[np.ndarray,np.ndarray]:
    """
    Preforms one single step forward in time

    Args:
        h (float): Time step size
        x (2xN array): Current particle positions (first row are horizontal coordinates, second row are vertical coordinates)
        v (2xN array): Current particle velocities (first row are horizontal coordinates, second row are vertical coordinates)
        part (dictionary['radius','spring']): Stores properties of the particle (radius r of all particles and spring constant of all particles)
        box (2x2 array): Array storing the corners of the box (np.vstack([low,upp]))
        g (float): Single number, the amount g of gravity
        N (Integer): Number of particles

    Returns:
        tuple: xnew(2xN array), vnew(2xN array) which are the updated position and velocity
    """
    #If N is not passed into the function:
    # N = len(x[0])

    #Set up forces array
    Forces = np.zeros((2,N))

    grid,gridArray,gridLength = gridBuilder(x,box,N)
    
    partNet = connect(grid,gridArray,N)

    Forces = forceWall(Forces,x,gridLength,box,part,N)

    Forces = forceParticle(Forces,x,partNet,part,N)

    xnew = x + (h * v) + (h**2)*Forces
    vnew =(xnew - x)/ h 

    return xnew, vnew


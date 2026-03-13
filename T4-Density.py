# Not sure if this is useful, made before seeing Isaac's version. Pretty much does the same as density_calc (I think)
# Added just in case

# How it works:
    # 'Creates' small box around each integer y value in box
        # length of each small box = length of large box (x)
        # width of each small box = tol * 2
    # density array = number of particles in each small box
    # Goes through each particle, checks if particle is inside any box
    # Adds 1 to density array if is (index = y value of small box)
    # Returns average density for each small box


# Returns list of average densities for each y
@njit
def get_density(N, Y, box):
    density = np.zeros(int(box[1, 1] - box[0, 1] + 1))

    tol = 0.25 # Tolerance can be changed
    
    for particle in range(N):
        # Particle's Y coordinate
        particle_Y = Y[particle]

        particle_Y_index = round(particle_Y)

        if abs(particle_Y_index - particle_Y) <= tol and box[0, 1] <= particle_Y_index <= box[1, 1]:
            density[particle_Y_index] += 1
     
    density /= (box[1, 0] * 2 * tol)

    return density
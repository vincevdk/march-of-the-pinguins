import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import importlib

from src.functions import *
from src.config import *
from src.allocating import*
from src.particles import *

import src.functions
#importlib.reload(src.functions)

#####initialize simulation######
fig = plt.figure()

#allocate memory
pos=allocating_variables()                                                     
radii = size_of_particles(N_particles, 2, 0.1)
colours_particles = np.zeros(shape=(N_particles,N_steps),dtype='object')
colours_orientation = np.zeros(shape=(N_particles, N_steps),dtype='object')
orientation = np.zeros((N_particles, N_steps))

# initialize variables
pos[:,:,0]=cubic_latice(N_particles)
colours_particles[:,:] = '-g' 
colours_orientation[:,:] = '-b'

orientation[:,0] = orientation_at_0(N_particles) # orientation in radians!
print(orientation[:,0])
###perform simulation
for time_step in range(1,N_steps):
    print("step is",time_step)
    neighbors, distance_matrix, distance_x, distance_y = check_for_neighbors(pos[:,:,time_step-1])
    displacement=sum_force(radii,distance_matrix,distance_x, distance_y)
#    torque = change_torque(bisection, orientation, neighbors)
#    orientation = update_orientation(torque,orientation,radii,bisection)
    pos[:,:,time_step] = np.add(displacement,pos[:,:,time_step-1])


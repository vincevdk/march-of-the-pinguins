import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import importlib
from matplotlib import collections, transforms

from src.functions import *
from src.config import *
from src.allocating import*
from src.particles import *

import src.functions
#importlib.reload(src.functions)

#####initialize simulation######
#allocate memory
pos=allocating_variables()                                                     
radii = size_of_particles(N_particles, 1, 0.1)
colours_particles = np.zeros(shape=(N_particles,N_steps),dtype='object')
colours_orientation = np.zeros(shape=(N_particles, N_steps),dtype='object')
orientation = np.zeros((N_particles, N_steps))
area = np.pi * (radii**2)*500

# initialize variables
pos[:,:,0]=cubic_latice(N_particles_x,N_particles_y)
colours_particles[:,:] = '-g' 
colours_orientation[:,:] = '-b'
orientation[:,0] = orientation_at_0(N_particles) # orientation in radians!

###perform simulation
for time_step in range(1,N_steps):
    pos[:,:,time_step], orientation[:,time_step], colours_particles[:,time_step],colours_orientation[:,time_step],displacement = update_position_and_orientation(
        pos[:,:,time_step-1], 
        orientation[:,time_step-1],
        radii,
        colours_particles[:,time_step-1], 
        colours_orientation[:,time_step-1])

# dataprocessing
orientation = np.radians(orientation)
radius = np.reshape(radii, (N_particles,1))
orientation_xy = np.zeros(shape=(N_particles,2,N_steps))
orientation_xy[:,0,:] = pos[:,0,:]+radius*np.cos(orientation)
orientation_xy[:,1,:] = pos[:,1,:]+radius*-np.sin(orientation)
area_circles = np.pi * np.power(radii,2)

# calculating observables
ACV=auto_correlation_velocity(pos,timestep=1)#what is the timestepsize?
MSD=mean_square_displacement(pos)

####create plot ####
lns = []
fig, ax = plt.subplots()
for steps in range (N_steps): # loop through timesteps
    col_circles = collections.EllipseCollection(1.5*radii,1.5*radii,
                                                np.zeros_like(radii),
                                                offsets=pos[:,:,steps], units='x',
                                                transOffset=ax.transData,
                                                facecolors='none',
                                                edgecolors=colours_particles[:,steps])

    f = np.hstack((pos[:,:,steps],orientation_xy[:,:,steps]))
    segments = f.reshape((N_particles, 2, 2))
    col_orientation = collections.LineCollection(segments,colors = colours_orientation[:,steps])
    im1=ax.add_collection(col_circles)
    im2=ax.add_collection(col_orientation)
    lns.append([im1,im2])

ani = animation.ArtistAnimation(fig, lns, interval=1000,blit=True)


ax.set_xlim(-2,5)
ax.set_ylim(-2,5)
plt.show()


import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.functions import *
from src.config import *
from src.allocating import*
from src.particles import *

#####initialize simulation######

fig = plt.figure()
pos=allocating_variables()                                                     
radii = size_of_particles(N_particles, 1, 0.1)
pos[:,:,0]=cubic_latice(N_particles)  
colours = np.zeros(shape=(N_particles,N_steps),dtype='object')
#colours = []
#print(len(pos))
colours[:,:] = '-g' 
print(colours)
###perform simulation
for time_step in range(1,N_steps):
    pos[:,:,time_step], colours[:,time_step] = update_position(pos[:,:,time_step-1],colours[:,time_step-1])


####create plot ####
lns = []
trans = plt.axes().transAxes
for steps in range (N_steps): # loop through timesteps
    x,y = particle(pos[:,0,steps], pos[:,1,steps], radii[:])
    lns_timestep = []
    for i in range(N_particles): #loop through particles (want to get rid of 
                                 # this, how can we do this?)
        ln1, = plt.plot(x[:,i], y[:,i], colours[i,steps], lw=2)
        ln2, = plt.plot([pos[i,0,steps],pos[i,0,steps]+radii[i]],[pos[i,1,0],pos[i,1,steps]], 'r-',lw=2)
        lns_timestep.append(ln1)
        lns_timestep.append(ln2)
    lns.append(lns_timestep)

plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.axes().set_aspect('equal')

ani = animation.ArtistAnimation(fig, lns, interval=50)
plt.show()









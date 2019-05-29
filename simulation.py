import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import importlib
from IPython.display import HTML

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
radii = size_of_particles(N_particles, 1, 0.1)
colours_particles = np.zeros(shape=(N_particles,N_steps),dtype='object')
colours_orientation = np.zeros(shape=(N_particles, N_steps),dtype='object')
orientation = np.zeros((N_particles, N_steps))

# initialize variables
pos[:,:,0]=cubic_latice(N_particles_x,N_particles_y)
colours_particles[:,:] = '-g' 
colours_orientation[:,:] = '-b'

orientation[:,0] = orientation_at_0(N_particles) # orientation in radians!
print(orientation[:,0])
###perform simulation
for time_step in range(1,N_steps):
    print("step is",time_step)
    pos[:,:,time_step], orientation[:,time_step], colours_particles[:,time_step],colours_orientation[:,time_step],displacement = update_position_and_orientation(
        pos[:,:,time_step-1], 
        orientation[:,time_step-1],
        radii,
        colours_particles[:,time_step-1], 
        colours_orientation[:,time_step-1],time_size)

# dataprocessing
orientation = np.radians(orientation)
radius = np.reshape(radii, (N_particles,1))
orientation_x = pos[:,0,:]+radius*np.cos(orientation)
orientation_y = pos[:,1,:]+radius*-np.sin(orientation)

# calculating observables
ACV=auto_correlation_velocity(pos,timestep=1)#what is the timestepsize?
MSD=mean_square_displacement(pos)
# diffusion stil needs testing
center_mass_in_time=center_of_mass(pos,radii,time_step)
Diffusion=func_diffusion(center_mass_in_time,time_size)

####create plot ####
lns = []
trans = plt.axes().transAxes
w = np.linspace(1,10,N_steps)
for steps in range (N_steps): # loop through timesteps
    x,y = particle(pos[:,0,steps], pos[:,1,steps], radii[:], N_particles)
    lns_timestep = []
    for i in range(N_particles): #loop through particles (want to get rid of 
                                 # this, how can we do this?)
        ln1, = plt.plot(x[:,i], y[:,i], colours_particles[i,steps], lw=2)

        ln2, = plt.plot([pos[i,0,steps],orientation_x[i,steps]],[pos[i,1,steps],orientation_y[i,steps]], colours_orientation[i,steps],lw=2)

        lns_timestep.append(ln1)
        lns_timestep.append(ln2)
        #lns_timestep.append(ln3)
    lns.append(lns_timestep)
    print("step animation is:",steps," out of ",N_steps)

plt.figure()

plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.axes().set_aspect('equal')

ani = animation.ArtistAnimation(fig, lns, interval=20/time_size,blit=True)
#ani.save("test.html")
HTML(ani.to_html5_video())

plt.show()


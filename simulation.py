import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import importlib

from src.functions import *
from src.config import *
from src.allocating import*
from src.particles import *

import src.functions
importlib.reload(src.functions)
#####initialize simulation######

fig = plt.figure()
pos=allocating_variables()                                                     
radii = size_of_particles(N_particles, 1, 0.1)
pos[:,:,0]=cubic_latice(N_particles)  
colours = np.zeros(shape=(N_particles,N_steps),dtype='object')
mid_angle = np.zeros((N_steps, N_particles))
deviation_0 = np.random.uniform(-0.785*180/np.pi, 0.785*180/np.pi, N_particles)
#deviation_0[deviation_0>0] = 360 - deviation_0[deviation_0>0]
delta_theta = np.zeros((N_steps, N_particles))
#colours = []
#print(len(pos))
colours[:,:] = '-g' 

###perform simulation
for time_step in range(1,N_steps):
    pos[:,:,time_step], colours[:,time_step], mid_angle[time_step,:] = update_position(pos[:,:,time_step-1],colours[:,time_step-1])
    delta_theta[time_step, :] = delta_thetas(mid_angle[time_step-1,:], deviation_0)
mid_angle[0] = mid_angle[1]
#print(deviation_0[0], 'deviation')
print(deviation_0, 'deviation')
print(delta_theta[2,:], 'delta_theta')
print(mid_angle[2,:], 'bisection')


####data mining of the simulation####
ACV=auto_correlation_velocity(pos,timestep=1)#what is the timestepsize?????
MSD=mean_square_displacement(pos)

####create plot ####
lns = []
trans = plt.axes().transAxes
w = np.linspace(1,10,N_steps)
init_orien_x, init_orien_y = orientation_at_0(N_particles, N_steps)
for steps in range (N_steps): # loop through timesteps
    x,y = particle(pos[:,0,steps], pos[:,1,steps], radii[:], N_particles)
    lns_timestep = []
    for i in range(N_particles): #loop through particles (want to get rid of 
                                 # this, how can we do this?)
        ln1, = plt.plot(x[:,i], y[:,i], colours[i,steps], lw=2)

        orientation_x = pos[i,0,steps]+radii[i]*np.cos(mid_angle[steps,i]*np.pi/180)
        orientation_y = pos[i,1,steps]+radii[i]*-np.sin(mid_angle[steps,i]*np.pi/180)
        
        ln2, = plt.plot([pos[i,0,steps],orientation_x],[pos[i,1,steps],orientation_y], 'r-',lw=2)
        #ln3, = plt.plot([pos[i,0,steps],init_orien_x[steps,i]+pos[i,0,steps]],[pos[i,1,steps],init_orien_y[steps,i]+pos[i,1,steps]], 'r-',lw=3)

        lns_timestep.append(ln1)
        lns_timestep.append(ln2)
        #lns_timestep.append(ln3)
    lns.append(lns_timestep)

plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.axes().set_aspect('equal')

ani = animation.ArtistAnimation(fig, lns, interval=50)
#HTML(ani.to_html5_video())


import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from IPython.display import HTML

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
mid_angle = np.zeros((N_steps, N_particles))
#colours = []
#print(len(pos))
colours[:,:] = '-g' 

###perform simulation
for time_step in range(1,N_steps):
    # force_per_particle, torque_per_particle,colours = calculate_force_and_torque(positions,orientations)

    # move_particle = (velocity * timestep) where speed is F/(alpha * Xi) 
    # turn_particle  = (angular velocity * timestep) where angular velocity is 
    
    

    pos[:,:,time_step], colours[:,time_step], mid_angle[time_step,:] = update_position(pos[:,:,time_step-1],colours[:,time_step-1])
mid_angle[0] = mid_angle[1]



####create plot ####
lns = []
trans = plt.axes().transAxes
w = np.linspace(1,10,N_steps)
for steps in range (N_steps): # loop through timesteps
    x,y = particle(pos[:,0,steps], pos[:,1,steps], radii[:])
    lns_timestep = []
    for i in range(N_particles): #loop through particles (want to get rid of 
                                 # this, how can we do this?)
        ln1, = plt.plot(x[:,i], y[:,i], colours[i,steps], lw=2)
        orientation_x = pos[i,0,steps]+radii[i]*np.cos(mid_angle[steps,i]*np.pi/180)
        orientation_y = pos[i,1,steps]+radii[i]*-np.sin(mid_angle[steps,i]*np.pi/180)
        
        original_orien_x = pos[i,0,steps]+radii[i]*np.cos(mid_angle[steps,i]*np.pi/180)
        original_orien_y = pos[i,1,steps]+radii[i]*-np.sin(mid_angle[steps,i]*np.pi/180)
        
        ln2, = plt.plot([pos[i,0,steps],orientation_x],[pos[i,1,steps],orientation_y], 'r-',lw=2)
        #ln4, = plt.plot([cycloid_c[steps], cycloid_x[steps]], [radii,cycloid_y[steps]], 'b-', lw=2)
        lns_timestep.append(ln1)
        lns_timestep.append(ln2)
    lns.append(lns_timestep)

plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.axes().set_aspect('equal')

ani = animation.ArtistAnimation(fig, lns, interval=50)
#HTML(ani.to_html5_video())
plt.show()




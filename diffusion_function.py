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
from src.dimensionless_scaling_test import *
import src.functions
from src.calculating_observables import *
#importlib.reload(src.functions)

time_size=1
N_steps=100

lambda_s2=[0.04,0.05,0.06,0.07,0.08]
lambda_a2=[0.1,0.14,0.2,0.3,0.45,0.67,1]
lambda_s2=[0.04]
lambda_a=0.2
#lambda_s2=0.05
#Diffusion=[]
for i in lambda_s2:
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
    
    scaling_force,scaling_torque,lambda_a,i,lambda_n,lambda_F_in,lambda_T_in=func_dimensionless_scaling(radii)
    
    
    
    orientation[:,0] = orientation_at_0(N_particles) # orientation in radians!
    ###perform simulation
    for time_step in range(1,N_steps):
        print("step is",time_step)
        pos[:,:,time_step], orientation[:,time_step], colours_particles[:,time_step],colours_orientation[:,time_step],displacement = update_position_and_orientation(
            pos[:,:,time_step-1], 
            orientation[:,time_step-1],
            radii,
            colours_particles[:,time_step-1], 
            colours_orientation[:,time_step-1],time_size,scaling_force,scaling_torque,lambda_a,lambda_s,lambda_n,lambda_F_in,lambda_T_in)
    
    # dataprocessing
#    orientation = np.radians(orientation)
#    radius = np.reshape(radii, (N_particles,1))
#    orientation_x = pos[:,0,:]+radius*np.cos(orientation)
#    orientation_y = pos[:,1,:]+radius*-np.sin(orientation)
    
    # calculating observables
#    ACV=auto_correlation_velocity(pos,timestep=1)#what is the timestepsize?
#    MSD=mean_square_displacement(pos)
    # diffusion stil needs testing
    center_mass_in_time=center_of_mass(pos,radii,time_step)
    b=func_diffusion(center_mass_in_time,time_size)
    Diffusion=np.append(Diffusion,[b])
    print("test okay")
    
    
#print(diffusion)

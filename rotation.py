import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import importlib
from simulation import perform_simulation
from src.calculating_observables import *
#from IPython.display import HTML

from src.functions import *
#from src.config import *
from src.allocating import*
from src.particles import *
from src.dimensionless_scaling_test import *
import src.functions

if __name__ == "__main__":
    ### Define inputs here
    lambda_a=0.20 # change for different behaviour
    lambda_s=0.08 # change for different behaviour

    #these are constant in all experiments 
    lambda_n=0.03
    lambda_F_in=0.3
    lambda_T_in=3

    #number of particles
    N_particles_x = 20
    N_particles_y = 10

    #number of timesteps
    N_steps = 10000
    time_size = 3
    lambda_array = [lambda_a, lambda_s, lambda_n, lambda_F_in, lambda_T_in]
    colours_particles, colours_orientation, orientation, pos,radii = perform_simulation(N_particles_x, 
                                                                                  N_particles_y,
                                                                                  lambda_array,
                                                                                  N_steps)
    N_particles = N_particles_x*N_particles_y
    radius = np.reshape(radii, (N_particles,1))

    orientation_x = pos[:,0,:]+radius*np.cos(orientation)
    orientation_y = pos[:,1,:]+radius*-np.sin(orientation)
           
    lns = []
    trans = plt.axes().transAxes
    w = np.linspace(1,10,N_steps)

    order_parameter = np.zeros((int(N_steps/100)))
    j=0
    ### create animation ###
    for steps in range(N_steps):
        if steps%300==0:
            #   for steps in range (N_steps): # loop through timesteps
            x,y = particle(pos[:,0,steps], pos[:,1,steps], radii[:], N_particles)
            lns_timestep = []
            for i in range(N_particles): #loop through particles (want to get rid of # this, how can we do this?)
                ln1, = plt.plot(x[:,i], y[:,i], colours_particles[i,steps], lw=2)

                ln2, = plt.plot([pos[i,0,steps],orientation_x[i,steps]],[pos[i,1,steps],orientation_y[i,steps]], colours_orientation[i,steps],lw=2)

                lns_timestep.append(ln1)
                lns_timestep.append(ln2)

            lns.append(lns_timestep)
        if steps%100==0:
            order_parameter[j] = orientational_order_parameter(orientation[:,steps-1])
            j += 1

        print("step animation is:",steps," out of ",N_steps)
    plt.figure()
    x = np.arange(j)*100
    plt.plot(x[30:],order_parameter[30:])
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$\phi$')

    fig = plt.figure()
    x = np.arange(j)*100
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.axes().set_aspect('equal')

    ani = animation.ArtistAnimation(fig, lns, interval=20/time_size,blit=True)

    #HTML(ani.to_html5_video())

    plt.show()


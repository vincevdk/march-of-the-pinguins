import numpy as np
from simulation import perform_simulation
from src.calculating_observables import *
import matplotlib.pyplot as plt
#from  orientational_order_parameter


### Define inputs here                                                                                                                               
lambda_a=[0.45, 0.30] # change for different behaviour                                                                                                  
lambda_s=[0.04, 0.04] # change for different behaviour                                                                                                       
#these are constant in all experiments                                                                                                               
lambda_n=0.03
lambda_F_in=0.3
lambda_T_in=3

#number of particles                                                                                                                                 
N_particles_x = np.array([5,10,10])
N_particles_y = np.array([10,10,20])
N_particles = N_particles_x * N_particles_y
print(N_particles)
#number of timesteps                                                                                                                                 
N_steps = 500
time_size = 3

MSD_array = np.zeros((len(N_particles_x), len(lambda_a))) 

for i in range(len(N_particles_x)):
    for j in range(len(lambda_a)):
        lambda_array = [lambda_a[j], lambda_s[j], lambda_n, lambda_F_in, lambda_T_in]
        colours_particles, colours_orientation, orientation, pos,radii = perform_simulation(N_particles_x[i],
                                                                                            N_particles_y[i],
                                                                                            lambda_array,
                                                                                            N_steps)
        MSD_array[i,j] = mean_square_displacement(pos[:,:,N_steps-1],pos[:,:,0])
    print(MSD_array)
#    print(MSD/(6*N_steps), N_particles_x, N_particles_y,'order')

plt.figure()
N_particles = N_particles_x * N_particles_y
plt.plot(N_particles, MSD_array)
plt.show()

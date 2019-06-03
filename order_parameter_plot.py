import numpy as np
from simulation import perform_simulation
from src.calculating_observables import *
#from  orientational_order_parameter


### Define inputs here                                                                                                                               
lambda_a=[0.14,0.20,0.30,0.45,0.67] # change for different behaviour                                                                                                  
lambda_s=[0.06,0.06,0.06,0.06,0.06] # change for different behaviour                                                                                                       
#these are constant in all experiments                                                                                                               
lambda_n=0.03
lambda_F_in=0.3
lambda_T_in=3

#number of particles                                                                                                                                 
N_particles_x = 10
N_particles_y = 5
#number of timesteps                                                                                                                                 
N_steps = 1000
time_size = 3

for i in range(len(lambda_a)):
    lambda_array = [lambda_a[i], lambda_s[i], lambda_n, lambda_F_in, lambda_T_in]
    colours_particles, colours_orientation, orientation, pos,radii = perform_simulation(N_particles_x,
                                                                                        N_particles_y,
                                                                                        lambda_array,
                                                                                        N_steps)
    
    order = orientational_order_parameter(orientation[:,N_steps-1])
    print(order,'order')


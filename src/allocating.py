#from src.config import *
import numpy as np

def cubic_latice(N_particles_x,N_particles_y):
    pos_at_0=np.array(np.meshgrid(range(np.int(N_particles_x)), range(np.int(N_particles_y)))).T.reshape(-1, 2)
    deviations = np.random.rand(N_particles_x*N_particles_y,2)*0.05
    pos_at_0 = pos_at_0*2 + deviations
    return(pos_at_0)

def orientation_at_0(N_particles):
    orientation = np.ones(N_particles)
    deviation = np.random.uniform(-45, 45, N_particles)
    return(deviation)

def allocating_variables(N_particles,N_steps):
    pos=np.zeros(shape=(N_particles,2,N_steps))
    return(pos)

def size_of_particles(N_particles, mean, standard_deviation):
    radii = (np.random.normal(mean,standard_deviation,N_particles))
    return(radii)

from src.config import *
import numpy as np

def cubic_latice(N_particles):
    pos_at_0=np.array(np.meshgrid(range(np.int(N_particles**.5)), range(np.int(N_particles**.5)))).T.reshape(-1, 2)
    deviations = np.random.rand(N_particles,2)*0.05
    pos_at_0 = pos_at_0*2 + deviations
    return(pos_at_0)

def orientation_at_0(N_particles, N_steps):
    orientation = np.ones(N_particles)
    deviation = np.random.uniform(-0.785, 0.785, N_particles)
    deviation = np.tile(deviation, (N_steps,1))
    orientation_x = orientation*np.cos(deviation)
    orientation_y = orientation*np.sin(deviation)
    return(orientation_x, orientation_y)

def allocating_variables():
    pos=np.zeros(shape=(N_particles,2,N_steps))

    return(pos)

def size_of_particles(N_particles, mean, standard_deviation):
    radii = (np.random.normal(mean,standard_deviation,N_particles))
    return(radii)

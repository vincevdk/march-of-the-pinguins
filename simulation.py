import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import importlib
#from IPython.display import HTML
import ffmpeg
from src.functions import *
#from src.config import *
from src.allocating import*
from src.particles import *
from src.calculating_observables import *
import src.functions
#importlib.reload(src.functions)
#plt.rcParams['animation.ffmpeg_path'] = '/opt/local/bin/ffmpeg'
#####initialize simulation######
fig = plt.figure()
print(ffmpeg,'ffmpeg')

# initialize variables
def perform_simulation(N_particles_x, N_particles_y, lambda_array, N_steps):
    time_size=3
    N_particles = N_particles_x*N_particles_y
    pos=allocating_variables(N_particles, N_steps)
    radii = size_of_particles(N_particles, 1, 0.1)
    colours_particles = np.zeros(shape=(N_particles,N_steps),dtype='object')
    colours_orientation = np.zeros(shape=(N_particles, N_steps),dtype='object')
    orientation = np.zeros((N_particles, N_steps))

    pos[:,:,0]=cubic_latice(N_particles_x,N_particles_y)
    colours_particles[:,:] = '-g' 
    colours_orientation[:,:] = '-b'
 
    orientation[:,0] = orientation_at_0(N_particles) # orientation in radians!
    
    ###perform simulation
    for time_step in range(1,N_steps):
        pos[:,:,time_step], orientation[:,time_step], colours_particles[:,time_step], colours_orientation[:,time_step], displacement = update_position_and_orientation(pos[:,:,time_step-1], 
                                                       orientation[:,time_step-1],
                                                       radii,
                                                       colours_particles[:,time_step-1], 
                                                       colours_orientation[:,time_step-1],
                                                       time_size, lambda_array)
    # dataprocessing
    orientation = np.radians(orientation)
    return(colours_particles,colours_orientation, orientation, pos, radii)

def calculate_ACV(pos,timestep):
    # calculating observables
    ACV=auto_correlation_velocity(pos,timestep=1)#what is the timestepsize?
    return(ACV)

def calculate_MSD(pos):
    MSD=mean_square_displacement(pos)
    return(MSD)

def calculate_diffusion(pos, radii, time_step, time_size):
    # diffusion stil needs testing
    center_mass_in_time=center_of_mass(pos,radii,time_step)
    Diffusion=func_diffusion(center_mass_in_time,time_size)
    return(Diffusion)


#################################
if __name__ == "__main__":
    ### Define inputs here
    lambda_a=0.05 # change for different behaviour
    lambda_s=0.08 # change for different behaviour

    #these are constant in all experiments 
    lambda_n=0.03
    lambda_F_in=0.3
    lambda_T_in=3

    #number of particles
    N_particles_x = 20
    N_particles_y = 10

    #number of timesteps
    N_steps = 300
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
        print("step animation is:",steps," out of ",N_steps)

    plt.figure()

    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.axes().set_aspect('equal')

    ani = animation.ArtistAnimation(fig, lns, interval=20/time_size,blit=True)
    FFwriter = animation.FFMpegWriter(fps=30, codec="libx264")     
    ani.save('basic_animation1.mp4', writer = FFwriter )
    #HTML(ani.to_html5_video())

    plt.show()


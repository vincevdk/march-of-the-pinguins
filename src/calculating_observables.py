import numpy as np

def auto_correlation_velocity(pos,timestep):
    auto_correlation=np.zeros(shape=len(pos[:,0,0]))
    diff=np.diff(pos)
    velocity=np.sqrt(np.sum((diff/timestep)**2,axis=1))
    for i in range(len(pos[:,0,0])):
        auto_correlation[i]=np.array(np.correlate(velocity[i,:],velocity[i,:]))/velocity[i,:].size
    return(auto_correlation)

def mean_square_displacement(pos):
    diff=np.diff(pos)
    sum_square=np.mean(diff**2,axis=(1,2))
    return(sum_square)

def orientational_order_parameter(orientation):
    order_parameter = np.mean(orientation, axis = 1)
    return(order_parameter)





def center_of_mass(pos,radii,time_step):
    time_step+=1
    mass_particles=np.pi*radii**2
    mass_pos=np.zeros(shape=(len(radii),2,time_step))
    
    for i in range(time_step):
        for j in range(len(radii)):
            for x in range(2):
                mass_pos[j,x,i]=mass_particles[j]*pos[j,x,i]
    
    center_mass_in_time=np.sum(mass_pos,axis=0)/np.sum(mass_particles)
    
    return(center_mass_in_time)
    
def func_diffusion(center_mass_in_time,time_size):
    VCMIT=np.diff(center_mass_in_time/time_size)
    mean_VCMIT=np.mean(VCMIT**2)
    
    for t in range(len(VCMIT)):
        cos_theta=(VCMIT[:,0]*VCMIT[:,t])/(np.abs(VCMIT[:,0])*np.abs(VCMIT[:,t]))
    Diffusion=1/2*mean_VCMIT*np.sum(cos_theta)
    return(Diffusion)
    

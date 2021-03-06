import numpy as np

def auto_correlation_velocity(pos,timestep):
    auto_correlation=np.zeros(shape=len(pos[:,0,0]))
    diff=np.diff(pos)
    velocity=np.sqrt(np.sum((diff/timestep)**2,axis=1))
    for i in range(len(pos[:,0,0])):
        auto_correlation[i]=np.array(np.correlate(velocity[i,:],velocity[i,:]))/velocity[i,:].size
    return(auto_correlation)

def mean_square_displacement(pos_at_t,pos_zero):
    msd = np.mean(np.power(pos_at_t-pos_zero,2))
    return(msd)

def orientational_order_parameter(orientation):
    order_parameter = np.sqrt(np.sum(np.cos(orientation))**2+np.sum(np.sin(orientation))**2)/len(orientation)
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
    #### needs works and unclear what goes wrong
      
    #This part calculates the velocity by calculating the difference between time steps
    VCMIT=np.diff(center_mass_in_time)

    #this function  allocates memory    
    cosine_array=np.zeros(shape=(VCMIT.shape[1]))

    # this functions calculates the cos(phi(t)), using vector calculas relationship for cos(x)=(a*b)/(sqrt(sum(a**2)))*sqrt(sum(b**2))))
    for t in range(VCMIT.shape[1]):
        cosine_array[t]=(np.sum(VCMIT[:,0]*VCMIT[:,t]))/(np.sqrt(np.sum(VCMIT[:,0]**2))*np.sqrt(np.sum(VCMIT[:,t]**2)))    
    
    ##this part of the code is wrong logical
#    a=0
#    for x in range(len(cosine_array)):
#        a+=np.mean(cosine_array[0:x])
    
    ## V_c is average velocity
    V_c=np.mean(np.sqrt(np.sum(VCMIT**2,axis=0)))
#    
    
    ## trying to get it to work with correlate function
    a=np.correlate(cosine_array,cosine_array,'full')
#    a=a[a.size/2:]
    
    #the formula Diffusion
    Diffusion=(1/2)*V_c**2*a
    

    return(Diffusion)
    

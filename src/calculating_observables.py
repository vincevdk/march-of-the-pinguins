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

import numpy as np
import matplotlib.pyplot as py


def cubic_latice(N_particles,L,W):
    pos_at_0=np.array(np.meshgrid(range(np.int(N_particles**.5)), range(np.int(N_particles**.5)))).T.reshape(-1, 2)
    pos_at_0[:,0]=pos_at_0[:,0]*W
    pos_at_0[:,1]=pos_at_0[:,1]*L

    return(pos_at_0)



def distance_and_direction(pos_at_t):
    a=np.tile(pos_at_t[:,0],(len(pos_at_t),1))
    at=np.transpose(a)

    distance_x=np.abs(at-a)
    
    b=np.tile(pos_at_t[:,1],(len(pos_at_t),1))
    bt=np.transpose(b)

    distance_y=np.abs(bt-b)
    
    
    distance=np.sqrt(distance_x**2+distance_y**2)
    
    
    
    
    return(distance)
    
    

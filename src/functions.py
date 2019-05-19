import numpy as np
import matplotlib.pyplot as py

def cubic_latice(N_particles):
    pos_at_0=np.array(np.meshgrid(range(np.int(N_particles**.5)), range(np.int(N_particles**.5)))).T.reshape(-1, 2)
    pos_at_0[:,0]=pos_at_0[:,0]*2
    pos_at_0[:,1]=pos_at_0[:,1]*2
    return(pos_at_0)

def size_of_particles(N_particles, mean, standard_deviation):
    radii = (np.random.normal(mean,standard_deviation,N_particles))
    return(radii)

def check_for_neighbors(pos,colours):
#    colours = ['-g']*len(pos)
    print(len(pos),'length')
    for i in range(len(pos)): # loop through particles
        distance_between_particles = ((pos[i,0]-pos[:,0])**2+(pos[i,0]-pos[:,1])**2)**0.5
        neighbors = np.asarray(distance_between_particles<2.7).nonzero()
        print(neighbors)
        if len(neighbors[0])  <= 4:
            colours[i] = '-b'
            print(colours)
    return(colours)
        
def update_position(position, colours):
    colours = check_for_neighbors(position,colours)
    position += 0.1
    return(position, colours)

def distance_and_direction(pos_at_t):
    a=np.tile(pos_at_t[:,0],(len(pos_at_t),1))
    at=np.transpose(a)

    distance_x=np.abs(at-a)
    
    b=np.tile(pos_at_t[:,1],(len(pos_at_t),1))
    bt=np.transpose(b)

    distance_y=np.abs(bt-b)
    
    
    distance=np.sqrt(distance_x**2+distance_y**2)    
    
    return(distance)
    
    

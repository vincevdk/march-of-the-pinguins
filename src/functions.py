import numpy as np
import matplotlib.pyplot as py
from math import acos

def cubic_latice(N_particles):
    pos_at_0=np.array(np.meshgrid(range(np.int(N_particles**.5)), range(np.int(N_particles**.5)))).T.reshape(-1, 2)
    deviations = np.random.rand(N_particles,2)*0.2
    pos_at_0 = pos_at_0*2 + deviations
    
    return(pos_at_0)

def size_of_particles(N_particles, mean, standard_deviation):
    radii = (np.random.normal(mean,standard_deviation,N_particles))
    return(radii)

def check_for_neighbors(pos,colours):

    for i in range(len(pos)): # loop through particles
        dis_particles = (np.power(pos[i,0]-pos[:,0],2)+np.power(pos[i,1]-pos[:,1],2))**0.5
        neighbors = np.asarray(np.logical_and(dis_particles<2.7,abs(dis_particles>0.1))).nonzero()
        angles = np.zeros(len(neighbors[0]))
        for j in range(len(neighbors[0])):

            difference_vector = [pos[neighbors[0][j],0]-pos[i,0],pos[neighbors[0][j],1] - pos[i,1]]

            angles[j] = (calculate_angle_wrt_x_axis(difference_vector))

        if (abs(np.amax(angles) - np.amin(angles)) <= 180):
            colours[i] = '-b'

    return(colours)
        
def update_position(position, colours):
    colours = check_for_neighbors(position,colours)
    position += 0.1
    return(position, colours)

def calculate_angle_wrt_x_axis(v):
    # see first answer: https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points

    norm_v = np.linalg.norm(v)
    
    cosx=v[0]/norm_v
    degrees = acos(cosx) * 180/np.pi
    print(degrees,'degrees')
    if v[1]<=0:
        return degrees
    else:
        return (360-degrees)
    
def circle_overlap(R1,R2,d):
    if d<=R1+R2 and np.abs(R1-R2)<=d:
        a=R1**2*np.arccos((d**2+R1**2-R2**2)/(2*d*R1))+R2**2*np.arccos((d**2-R1**2+R2**2)/(2*d*R2))-(1/2)*np.sqrt((-d+R1+R2)*(d+R1-R2)*(d-R1+R2)*(d+R1+R2))
    if np.abs(R1-R2)>d:
        a=min([R1 ,R2])**2*np.pi
    return(a)

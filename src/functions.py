import numpy as np
import matplotlib.pyplot as py
from math import acos

def cubic_latice(N_particles):
    pos_at_0=np.array(np.meshgrid(range(np.int(N_particles**.5)), range(np.int(N_particles**.5)))).T.reshape(-1, 2)
    deviations = np.random.rand(N_particles,2)*0.05
    pos_at_0 = pos_at_0*2 + deviations
    
    return(pos_at_0)

#def orientation_at_0():

def size_of_particles(N_particles, mean, standard_deviation):
    radii = (np.random.normal(mean,standard_deviation,N_particles))
    return(radii)

def check_for_neighbors(pos,colours):
    bisection = np.zeros(len(pos))
    for i in range(len(pos)): # loop through particles
        dis_particles = (np.power(pos[i,0]-pos[:,0],2)+np.power(pos[i,1]-pos[:,1],2))**0.5
        neighbors = np.array(np.logical_and(dis_particles<2.7,abs(dis_particles>0.1))).nonzero()
        angles = np.zeros(shape=(len(neighbors[0]),2))
        for j in range(len(neighbors[0])):

            difference_vector = [pos[neighbors[0][j],0]-pos[i,0],pos[neighbors[0][j],1] - pos[i,1]]

            angles[j,0] = (calculate_angle_wrt_pos_x_axis(difference_vector))
            angles[j,1] = (calculate_angle_wrt_neg_x_axis(difference_vector))
            
        if i == 0 or i == 4 or i == 20 or i == 24:
            colours[i] = '-y'
        
        if (abs(np.amax(angles[:,0]) - np.amin(angles[:,0])) <= 180):
            bisection[i] = (max(angles[:,0])+min(angles[:,0]))/2
            colours[i] = '-b'
        
        elif (abs(np.amax(angles[:,1]) - np.amin(angles[:,1])) <= 180):
            print((max(angles[:,1])+min(angles[:,1]))/2)
            bisection[i] = (max(angles[:,1])+min(angles[:,1]))/2+180
            colours[i] = '-r'

#        else:
#            bisection[i] = 90       
    return(colours, bisection)
        
def update_position(position, colours):
    colours, bisection = check_for_neighbors(position,colours)
    position += 0.1
    return(position, colours, bisection)

def calculate_angle_wrt_pos_x_axis(v):
    # see first answer: https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points

    norm_v = np.linalg.norm(v)
    
    cosx=v[0]/norm_v
    degrees = acos(cosx) * 180/np.pi

    if v[1]<=0:
        return degrees
    else:
        return (360-degrees)
    
def calculate_angle_wrt_neg_x_axis(v):
    # see first answer: https://stackoverflow.com/questions/31735499/calculate-\angle-clockwise-between-two-points                                              

    norm_v = np.linalg.norm(v)

    cosx=-v[0]/norm_v
    degrees = acos(cosx) * 180/np.pi

    if v[1]<=0:
        return degrees
    else:
        return (360-degrees)

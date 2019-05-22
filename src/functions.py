import numpy as np
import matplotlib.pyplot as py
from math import acos
import matplotlib.colors as colorlib


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

    
def update_color(radii,pos,colours):
    Blues = py.get_cmap('Blues')
    overlap=np.zeros(len(pos))
    percentage_overlap=np.zeros(len(pos))

    for i in range(len(pos)): # loop through particles
        
        dis_particles = (np.power(pos[i,0]-pos[:,0],2)+np.power(pos[i,1]-pos[:,1],2))**0.5
        neighbors = np.array(np.logical_and(dis_particles<2.7,abs(dis_particles>0.1))).nonzero()
        
        for j in neighbors[0]:
            
            d=(np.power(pos[i,0]-pos[j,0],2)+np.power(pos[i,1]-pos[j,1],2))**0.5
            overlap[i]+=circle_overlap(radii[i],radii[j],d)
            percentage_overlap[i]=overlap[i]/(np.pi*radii[i])
        if percentage_overlap[i]==0:
            colours[i] = '-r'
        else:
 
            colours[i]=colorlib.to_hex( Blues(percentage_overlap[i]*10+0.5)[0:3])
    test_overlap=overlap
    return(colours)

def circle_overlap(R1,R2,d):
    if d<=R1+R2 and np.abs(R1-R2)<=d:
        a=R1**2*np.arccos((d**2+R1**2-R2**2)/(2*d*R1))+R2**2*np.arccos((d**2-R1**2+R2**2)/(2*d*R2))-(1/2)*np.sqrt((-d+R1+R2)*(d+R1-R2)*(d-R1+R2)*(d+R1+R2))
    if np.abs(R1-R2)>d:
        a=min([R1 ,R2])**2*np.pi
    if np.abs(R1+R2)<d:
        a=0
    return(a)
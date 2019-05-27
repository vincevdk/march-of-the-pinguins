import numpy as np
import matplotlib.pyplot as py
from math import acos
import matplotlib.colors as colorlib
from src.config import *

def update_position_and_orientation(position, 
                                    orientation, 
                                    radii, 
                                    colours_particles, 
                                    colours_orientation):
    bisection = np.zeros(len(position))
    neighbors, distance_matrix, distance_x, distance_y = check_for_neighbors(position)
    bisection,colours_orientation = calculate_bisection(position, 
                                    neighbors, 
                                    distance_matrix, 
                                    distance_x, 
                                    distance_y,
                                    colours_orientation)
    displacement=1
#    displacement,area=sum_force(radii,distance_matrix,distance_x, distance_y)
    torque = change_torque(bisection, orientation, neighbors)
    orientation = update_orientation(torque,orientation,radii,bisection)
<<<<<<< HEAD
    position = np.add(displacement*.1,position)+1
#    colours_particles = update_color(radii,area,colours_particles)
    return(position, orientation, colours_particles, colours_orientation,displacement)

def sum_force(radii,d,distance_x, distance_y):
    force_x=np.zeros(shape=len(radii))
    force_y=np.zeros(shape=len(radii))
    displacement=np.zeros(shape=(len(radii),2))

    force_x,force_y,area=repulsion_force(radii,d,distance_x, distance_y)
    displacement[:,0]=force_x
    displacement[:,1]=force_y

    
    return(displacement,area)
=======
#    force = change_force(orientation, neighbors, bisection)
#    position = update_position(position, force, radii)
    colours = update_color(radii,position,colours_particles)
    return(position, orientation, colours_particles, colours_orientation)
>>>>>>> 50c22a317274643b9d096c3852009e9011a0e6c9

def check_for_neighbors(pos_at_t):
    """Make a N by N matrix and calculate distances between all particles 
    """
    a=np.tile(pos_at_t[:,0],(len(pos_at_t),1))
    at=np.transpose(a)
    distance_x=np.array(at-a)
    b=np.tile(pos_at_t[:,1],(len(pos_at_t),1))
    bt=np.transpose(b)
    distance_y=np.array(bt-b)
    distance=np.array(np.sqrt(distance_x**2+distance_y**2))
    neighbors = np.array(np.logical_and(np.abs(distance)<2.7, np.abs(distance)>0.1))
    return(neighbors, distance, distance_x, distance_y)

def delta_thetas(bisection, orientation):
    delta_theta = np.zeros(len(bisection))
    non_zero_bisections = bisection.nonzero()

    for j in non_zero_bisections[0]:
        delta_theta[j] +=  ((bisection[j] - orientation[j] + 180) % 360 - 180)
    return(delta_theta)

def update_position(position, force,radii):
    print(force,'force')
    speed = force/(radii**2)
    position += speed # delta t = 1
    return(position)
    

def change_force(orientation, neighbors, bisection):
    orientation_x
    orientation_y
    vector_inward_force = np.zeros((N_particles,2))
    scalar_inward_force = magnitude_inward_force(neighbors, bisection)
    vector_inward_force[:,0] = scalar_inward_force(neighbors,bisection)*orientation_x
    vector_inward_force[:,0] = scalar_inward_force(neighbors,bisection)*orientation_y
    return(new_force)

def magnitude_inward_force(neighbors, bisection):
    return(bisection.nonzero())

def noise_torque(N_particles):
    noise = np.sqrt(0.03)*np.random.uniform(1, -1, N_particles)
    return(noise)

def boundary_torque(bisection,orientation):
    print(delta_thetas(bisection,orientation),'nonzero??')
    return(3 * delta_thetas(bisection, orientation))

def align_torque(orientation, neighbors):
    orientation_mismatch = np.zeros(len(orientation))
    for i in range(len(orientation)):
        neighbor_orientations = (neighbors[i]*orientation).nonzero()
        for j in neighbor_orientations[0]:
            
            orientation_mismatch[i] +=  ((orientation[j] - orientation[i] + 180) % 360 - 180)
    torque = 0.1 * orientation_mismatch 
    return(torque)

def change_torque(bisection, orientation, neighbors):
    # lamdba_Tin = 3
    # lamda_n = 0.03    
    # lamda_a = 0.1
    new_torque = boundary_torque(bisection, orientation)
#   new_torque = align_torque(orientation, neighbors) + noise_torque(N_particles)  + boundary_torque(bisection, orientation)
    return(new_torque)

def update_orientation(torque, orientation,radii, bisection):
    alpha_i = radii*2
    angular_velocity = torque/alpha_i
    orientation += angular_velocity
    return(orientation)

def calculate_bisection(pos,distance, neighbors, dis_x, dis_y, colour_orientation):
    angles_clockwise = np.zeros(shape=(N_particles,N_particles,2))
    angles_anticlockwise = np.zeros(shape=(N_particles,N_particles,))
    nearest_distance = distance*neighbors
    angles_clockwise = calculate_angle_wrt_pos_x_axis(nearest_distance, dis_x,dis_y)
    angles_anticlockwise = calculate_angle_wrt_neg_x_axis(nearest_distance, dis_x, dis_y)
    bisection = np.zeros((N_particles))

    for j in range(N_particles):
        if (abs(np.amax(angles_clockwise[j,:]) - np.amin(
                    angles_clockwise[j,np.nonzero(angles_clockwise[j,:])])) <= 180):
            bisection[j] = (np.amax(angles_clockwise[j,:])+np.amin(
                    angles_clockwise[j,np.nonzero(angles_clockwise[j,:])]))/2+180 
            colour_orientation[j] = '-r'
        elif (abs(np.amax(angles_anticlockwise[j,:]) - np.amin(
                    angles_anticlockwise[j,np.nonzero(angles_anticlockwise[j,:])])) <= 180):
           bisection[j] = -(np.amax(angles_anticlockwise[j,:])+np.amin(
                   angles_anticlockwise[j,np.nonzero(angles_anticlockwise[j,:])]))/2
           colour_orientation[j] = '-r'
    return(bisection,colour_orientation)

def calculate_angle_wrt_pos_x_axis(v, dis_x,dis_y):
    # see first answer: https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
    cosx = np.divide(dis_x, v, out=np.ones_like(dis_x), where=v!=0)
    degrees = np.arccos(cosx)*180/np.pi
    degrees = np.where(degrees!=0, np.where(dis_y<0,degrees,360-degrees),0)
    return(degrees)
    
def calculate_angle_wrt_neg_x_axis(v,dis_x,dis_y):
    # see first answer: https://stackoverflow.com/questions/31735499/calculate-\angle-clockwise-between-two-points                                            
    cosx = np.divide(-dis_x, v, out=np.ones_like(dis_x), where=v!=0)
    degrees = np.arccos(cosx) * 180/np.pi
    degrees = np.where(degrees!=0, np.where(dis_y<0,degrees,360-degrees),0)
    return(degrees)
    
def update_color(radii,area,colours):
    Blues = py.get_cmap('Blues')
    surface_area_circle=np.pi*radii**2
    
    percentage_overlap=np.sum(area,axis=1)/surface_area_circle
    

    
#    colours=np.where(percentage_overlap<=0.1,'-r',colorlib.to_hex( Blues(min(percentage_overlap,256))[0:3]))
    
    for i in range(len(radii)):
        if percentage_overlap[i]<0.1:
            colours[i] = '-r'
        else: 
            colours[i]=colorlib.to_hex( Blues(min(percentage_overlap[i],256))[0:3])
        
    
    
    
#    for i in range(len(pos)): # loop through particles
#        
#        dis_particles = (np.power(pos[i,0]-pos[:,0],2)+np.power(pos[i,1]-pos[:,1],2))**0.5
#        neighbors = np.array(np.logical_and(dis_particles<2.7,abs(dis_particles>0.1))).nonzero()
#        for j in neighbors[0]:
#            d=(np.power(pos[i,0]-pos[j,0],2)+np.power(pos[i,1]-pos[j,1],2))**0.5
#            overlap[i]+=circle_overlap(radii[i],radii[j],d)
#            percentage_overlap[i]=overlap[i]/(np.pi*radii[i])
#        if percentage_overlap[i]==0:
#            colours[i] = '-r'
#        else: 
#            colours[i]=colorlib.to_hex( Blues(percentage_overlap[i]*10+0.3)[0:3])
    return(colours)

def circle_overlap(R1,R2,d):
    if d<=R1+R2 and np.abs(R1-R2)<=d:
        a=R1**2*np.arccos((d**2+R1**2-R2**2)/(2*d*R1))+R2**2*np.arccos((d**2-R1**2+R2**2)/(2*d*R2))-(1/2)*np.sqrt((-d+R1+R2)*(d+R1-R2)*(d-R1+R2)*(d+R1+R2))
    elif np.abs(R1-R2)>d:
        a=min([R1 ,R2])**2*np.pi
    elif np.abs(R1+R2)<d:
        a=0
    return(a)
    
def repulsion_force(radii,d,distance_x, distance_y):
    R2,R1=np.meshgrid(radii,radii)
    d=np.abs(d)
    a=np.zeros(shape=(len(radii),len(radii)))
    
    a+=np.where(np.logical_and(np.logical_and(d<=R1+R2 , np.abs(R1-R2)<=d),d>0.10),R1**2*np.arccos((d**2+R1**2-R2**2)/(2*d*R1))+R2**2*np.arccos((d**2-R1**2+R2**2)/(2*d*R2))-(1/2)*np.sqrt((-d+R1+R2)*(d+R1-R2)*(d-R1+R2)*(d+R1+R2)),0)
    
    a+=np.where(np.logical_and((R1-R2)>d,d>0.1),R1**2*np.pi,0)
    a+=np.where(np.logical_and((R2-R1)>d,d>0.1),R2**2*np.pi,0)
    
    force_x=np.zeros(shape=(len(radii),len(radii)))
    force_y=np.zeros(shape=(len(radii),len(radii)))
    
    force_x=np.sum(np.where(np.logical_and(a>0.1,d>0.1),a*distance_x/d,0),axis=1)
    force_y=np.sum(np.where(np.logical_and(a>0.1,d>0.1),a*distance_y/d,0),axis=1)

    
    return(force_x,force_y,a)
    
def force_overlap(radii,pos,F_overlap_constant):
    Force_overlap=np.zeros(shape=(len(pos),2))
    for i in range(len(pos)): # loop through particles       
        dis_particles = (np.power(pos[i,0]-pos[:,0],2)+np.power(pos[i,1]-pos[:,1],2))**0.5
        neighbors = np.array(np.logical_and(dis_particles<2.7,abs(dis_particles>0.1))).nonzero()
        for j in neighbors[0]:
            
            d=(np.power(pos[i,0]-pos[j,0],2)+np.power(pos[i,1]-pos[j,1],2))**0.5
            direction=[pos[i,0]-pos[j,0],pos[i,1]-pos[j,1]]/d
            Force_overlap[i,:]=direction*circle_overlap(radii[i],radii[j],d)*F_overlap_constant
          
    return(Force_overlap)

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



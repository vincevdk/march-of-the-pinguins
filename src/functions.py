import numpy as np
import matplotlib.pyplot as py
from math import acos
import matplotlib.colors as colorlib


def cubic_latice(N_particles):
    pos_at_0=np.array(np.meshgrid(range(np.int(N_particles**.5)), range(np.int(N_particles**.5)))).T.reshape(-1, 2)
    deviations = np.random.rand(N_particles,2)*0.05
    pos_at_0 = pos_at_0*2 + deviations
    
    return(pos_at_0)

def orientation_at_0(N_particles, N_steps):
    orientation = np.ones(N_particles)
    deviation = np.random.uniform(-0.785, 0.785, N_particles)
    deviation = np.tile(deviation, (N_steps,1))
    orientation_x = orientation*np.cos(deviation)
    orientation_y = orientation*np.sin(deviation)
    return(orientation_x, orientation_y)

def delta_thetas(bisection, deviation_0):
    bisection[bisection<0] = 360 - abs(bisection[bisection<0])
    delta_theta = abs(360 - bisection - deviation_0)
    delta_theta[delta_theta>180] = 360 - delta_theta[delta_theta>180]
    delta_theta[bisection==0] = 0
    #delta_theta = delta_theta/360
    return(delta_theta)

def noise_torque(N_particles):
    noise = np.random.uniform(1, -1, N_particles)
    return(noise)

def change_torque():
    new_torque = noise_torque(N_particles) + delta_thetas(bisection, deviation_0)
    return(new_torque)

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
            bisection[i] = -(max(angles[:,1])+min(angles[:,1]))/2+180
            colours[i] = '-r'
                  
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
 
            colours[i]=colorlib.to_hex( Blues(percentage_overlap[i]*10+0.3)[0:3])
    return(colours)

def circle_overlap(R1,R2,d):
    if d<=R1+R2 and np.abs(R1-R2)<=d:
        a=R1**2*np.arccos((d**2+R1**2-R2**2)/(2*d*R1))+R2**2*np.arccos((d**2-R1**2+R2**2)/(2*d*R2))-(1/2)*np.sqrt((-d+R1+R2)*(d+R1-R2)*(d-R1+R2)*(d+R1+R2))
    if np.abs(R1-R2)>d:
        a=min([R1 ,R2])**2*np.pi
    if np.abs(R1+R2)<d:
        a=0
    return(a)

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

def Test_check_neighbours(pos_at_t):
    ### Make a N by N matrix and calculate distances between all particles
    a=np.tile(pos_at_t[:,0],(len(pos_at_t),1))
    at=np.transpose(a)
    
    
    distance_x=np.array(at-a)
    
    b=np.tile(pos_at_t[:,1],(len(pos_at_t),1))
    bt=np.transpose(b)

    distance_y=np.array(bt-b)
    
    
    distance=np.array(np.sqrt(distance_x**2+distance_y**2))
    #### make a True False Matrix and use it as a Mask
    neighbors =np.array( np.logical_and(np.abs(distance)<2.7, np.abs(distance)>0.1))
    c=np.where(neighbors==True)
    print(np.column_stack(c))
    nearest_distance=np.ma.array(distance,mask=~neighbors)
    
    
    distance_x=np.ma.array(distance_x,mask=~neighbors)
    distance_y=np.ma.array(distance_y,mask=~neighbors)
    unit_vector_distance=np.array([distance_x,distance_y]/distance)
    
    return(neighbors,distance,nearest_distance,unit_vector_distance)
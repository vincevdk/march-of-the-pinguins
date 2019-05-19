import numpy as np

def particle(a, b, r):
    # (a,b): the center of the circle
    # r: the radius of the circle 
    # T: The number of the segments                                   
    T = 100
    x = np.zeros((len(r),T))
    y =  np.zeros((len(r),T))
    theta = (np.linspace(0,2*np.pi,T))
    theta = np.tile(theta,(25,1)).T
    x = a + np.multiply(r,np.cos(theta))
    y = b + np.multiply(r,np.sin(theta))
    return x, y

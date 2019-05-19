import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.functions import *
from src.config import *
from src.allocating import*
from src.particles import *
#####initialize simulation######

fig = plt.figure()
pos=allocating_variables()                                                     
pos[:,:,0]=cubic_latice(N_particles,L,W)  
R = 0.5





####create plot####
lns = []
trans = plt.axes().transAxes

for i in range(N_particles):
    x,y = particle(pos[i,0,0], pos[i,1,0], R)
    ln1, = plt.plot(x, y, 'g-', lw=2)
    ln2, = plt.plot([pos[i,0,0],pos[i,0,0]+R],[pos[i,1,0],pos[i,1,0]+R], 'r-',lw=2)
    lns.append(ln1)
    lns.append(ln2)

ani = animation.ArtistAnimation(fig, lns, interval=50)
plt.show()

plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.axes().set_aspect('equal')








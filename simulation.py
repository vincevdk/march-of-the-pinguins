
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt


from src.functions import *
from src.config import *
from src.allocating import*


#####initialize simulation######
pos=allocating_variables()
pos[:,:,0]=cubic_latice(N_particles,L,W)




plt.figure("1")
plt.title(str(N_particles)+" particles in a box in a box")
plt.scatter(pos[:,0,-1],pos[:,1,-1])
plt.xlabel('x')
plt.ylabel('y')
plt.show()




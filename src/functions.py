import numpy as np
import matplotlib.pyplot as py


def fcc_lattice(N_particles,L):
    pos_at_0=np.array(np.meshgrid(range(N), range(N))).T.reshape(-1, 2)
    return(pos_at_0)





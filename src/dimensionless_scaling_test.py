import numpy as np



def func_dimensionless_scaling(radii):
    
    
    #normalized radius alpha
    normalized_radii=radii/1 # everything is scaled around 1
    alpha=normalized_radii
    
    #scaling dimensionless
    xi=1 
    #    xi=(32/3)*eta
    eta=3/32
    
    
    chi=1
    #    chi=4*np.pi*eta_r
    eta_r=1/(4*np.pi)
    
    
    #scaling force -> velocity
    scaling_force=1/(alpha*xi)
    scaling_torque=1/(alpha**2*chi)
    
    #list lambda
    
    #these are changing througout tests
    lambda_a=0.2 #[0.1,1]
    lambda_s=0.08 #[0.04,0.08]

    #these are constant in all experiments
    lambda_n=0.03
    lambda_F_in=0.3
    lambda_T_in=3
    
    #lambda_x=tau/tau_x

    
    return(scaling_force,scaling_torque,lambda_a,lambda_s,lambda_n,lambda_F_in,lambda_T_in)
import numpy as np
import matplotlib.pyplot as plt
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

def q(sigma_prev, steeringAngle, pedalSpeed, B, r, dt): # dynamics equation
    N = sigma_prev.shape[1]
    xi = np.zeros((5,N))
    for i in range(N):
        x = sigma_prev[0,i]
        y = sigma_prev[1,i]
        theta = sigma_prev[2,i]
        vel = sigma_prev[3,i]
        gamma = sigma_prev[4,i]
        v = sigma_prev[5,i]

        xi[0,i] = x + dt*vel*np.cos(theta)
        xi[1,i] = y + dt*vel*np.sin(theta)
        xi[2,i] = theta + dt*vel*np.tan(gamma)/B
        xi[3,i] = 5*r*pedalSpeed + v
        xi[4,i] = steeringAngle

    return xi
    # 8 by 16 matrix

def h(sigma_xp, B): # measurement equation
    N = sigma_xp.shape[1]
    sigma_z = np.zeros((2,N))
    for i in range(N):
        x = sigma_xp[0,i]
        y = sigma_xp[1,i]
        theta = sigma_xp[2,i]
        
        sigma_z[0,i] = x + 0.5*B*np.cos(theta) # measurement for x
        sigma_z[1,i] = y + 0.5*B*np.sin(theta) # measurement for y
    
    return sigma_z



def estRun(time, dt, internalStateIn, steeringAngle, pedalSpeed, measurement):
    # In this function you implement your estimator. The function arguments
    # are:
    #  time: current time in [s] 
    #  dt: current time step [s]
    #  internalStateIn: the estimator internal state, definition up to you. 
    #  steeringAngle: the steering angle of the bike, gamma, [rad] 
    #  pedalSpeed: the rotational speed of the pedal, omega, [rad/s] 
    #  measurement: the position measurement valid at the current time step
    #
    # Note: the measurement is a 2D vector, of x-y position measurement.
    #  The measurement sensor may fail to return data, in which case the
    #  measurement is given as NaN (not a number).
    #
    # The function has four outputs:
    #  x: your current best estimate for the bicycle's x-position
    #  y: your current best estimate for the bicycle's y-position
    #  theta: your current best estimate for the bicycle's rotation theta
    #  internalState: the estimator's internal state, in a format that can be understood by the next call to this function

    # Example code only, you'll want to heavily modify this.
    # this internal state needs to correspond to your init function:

    ################## MY CODE STARTS ##########################

    # parameters
    B = 0.8
    r = 0.425
    v = 0
    N = 8
    Swv = np.array([[v,       0.       , 0.        ],
                    [0,       1.0880701, 0.        ],
                    [0,       0.       , 2.98447239]])

    # storing x(k-1)
    x_prev = internalStateIn[0]
    y_prev = internalStateIn[1]
    theta_prev = internalStateIn[2]
    vel_prev = internalStateIn[3]
    gamma_prev = internalStateIn[4]
    Pm_prev = internalStateIn[5]

        ### 1. Prior Step
    # define auxiliary variable and its variance matrix
    xi_prev = np.array([[x_prev],
                        [y_prev],
                        [theta_prev],
                        [vel_prev],
                        [gamma_prev],
                        [0],
                        [0],
                        [0]])
    var1 = np.concatenate((Pm_prev, np.zeros((5,3))), axis=1)
    var2 = np.concatenate((np.zeros((3,5)), Swv), axis=1)
    var_xi_prev = np.concatenate((var1, var2), axis=0)

    # square root of (N * variance matrix)
    evalues, evectors = np.linalg.eig(8*var_xi_prev)
    assert (evalues >= 0).all()
    sqrt_var = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)

    # generate 2N sigma points
    sigma_prev = np.zeros((8,16))
    for i in range(0,8):
        sigma_prev[:,i] = xi_prev.reshape(8,) + sqrt_var[:,i]
    for i in range(8,16):
        sigma_prev[:,i] = xi_prev.reshape(8,) - sqrt_var[:,i-N]

    # compute the prior sigma points with dynamics equation
    sigma_xp = q(sigma_prev, steeringAngle, pedalSpeed, B, r, dt)
    #print(sigma_xp[2,:])
    
    # compute prior statistics
    x_est = np.mean(sigma_xp, axis=1)

    #Pm = np.cov(sigma_xp)
    Pm = np.zeros((5,5))
    for i in range(16):
        Pm += (sigma_xp[:,i] - x_est).reshape(5,1) @ (sigma_xp[:,i] - x_est).reshape(1,5)
    Pm = Pm / 16

    Pm = np.cov(sigma_xp)

    #Pm = np.zeros(5)
    #for i in range(16):
    #    Pm += np.square(sigma_xp[:,i] - x_est)
    #Pm = np.diag(Pm) / 16

    # Pm2 = np.cov()

    if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
        # have a valid measurement => Measurement update
        # calculate sigma_z points with measurment equation
        sigma_z = h(sigma_xp, B)

        # compute z_hat, Pzz, Pxz from sigma_z points
        z_est = np.mean(sigma_z, axis=1)

        #Pzz = np.zeros((2,2))
        #for i in range(16):
        #    Pzz += (sigma_z[:,i] - z_est).reshape(2,1) @ (sigma_z[:,i] - z_est).reshape(1,2)
        #Pzz = Pzz / 16
        Pzz = np.cov(sigma_z) + np.eye(2) * 1e-6
        #print(Pzz)

        Pxz = np.zeros((5,2))
        for i in range(16):
            Pxz += (sigma_xp[:,i] - x_est).reshape(5,1) @ (sigma_z[:,i] - z_est).reshape(1,2)
        Pxz = Pxz / 16

        # apply kalman filter
        K = Pxz @ np.linalg.inv(Pzz) # 5 by 2
        z = np.array([[measurement[0]],
                      [measurement[1]]])
        x_est = x_est.reshape(5,1) + K@(z - z_est.reshape(2,1)) 
        Pm = Pm - K@Pzz@K.T

    x = np.squeeze(x_est[0])
    y = np.squeeze(x_est[1])
    theta = np.squeeze(x_est[2])
    vel = np.squeeze(x_est[3])
    gamma = np.squeeze(x_est[4])

    #print(x)

    ################## MY CODE ENDS ##########################

    #### OUTPUTS ####
    # Update the internal state (will be passed as an argument to the function
    # at next run), must obviously be compatible with the format of
    # internalStateIn:
    internalStateOut = [x,
                     y,
                     theta,
                     vel,
                     gamma,
                     Pm]

    # DO NOT MODIFY THE OUTPUT FORMAT:
    return x, y, theta, internalStateOut 



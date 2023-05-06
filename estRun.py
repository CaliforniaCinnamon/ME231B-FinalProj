import numpy as np
import scipy
import matplotlib.pyplot as plt
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

def q(sigma_xi, steeringAngle, pedalSpeed, B, r, dt): # dynamics equation
    N = sigma_xi.shape[1]
    x_N = 3 # dimension of the state
    xi = np.zeros((x_N,N))
    for i in range(N):
        x = sigma_xi[0,i]
        y = sigma_xi[1,i]
        theta = sigma_xi[2,i]
        v1 = sigma_xi[3,i]
        v2 = sigma_xi[4,i]

        vel = (5*r*pedalSpeed) + v1
        gamma = steeringAngle + v2
        xi[0,i] = x + dt*vel*np.cos(theta)
        xi[1,i] = y + dt*vel*np.sin(theta)
        xi[2,i] = theta + dt*vel*np.tan(gamma)/B

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

def implement_UKF(dt, internalStateIn, steeringAngle, pedalSpeed, measurement, B, r):
    # tuning parameters: adjust them to make the best output
    v1 = 0.02
    v2 = 0.01
    Svw = np.array([[v1,      0,       0.       , 0.       ],
                    [0,       v2,      0.       , 0.       ],
                    [0,       0,       1.0880701, 0.        ],
                    [0,       0,       0.       , 2.98447239]])

    # storing x(k-1)
    x_prev = internalStateIn[0]
    y_prev = internalStateIn[1]
    theta_prev = internalStateIn[2]
    Pm_prev = internalStateIn[3]

        ### 1. Prior Step
    # define auxiliary variable (xi) and its variance matrix
    xi_prev = np.array([[x_prev],
                        [y_prev],
                        [theta_prev],
                        [0],
                        [0],
                        [0],
                        [0]])
    x_N = 3 # dimension of the state (x, y, theta)
    xi_N = 7 # dimension of xi (3+2+2=9 -- state, v, w respectively)
    var1 = np.concatenate((Pm_prev, np.zeros((x_N, xi_N - x_N))), axis=1)
    var2 = np.concatenate((np.zeros((xi_N - x_N, x_N)), Svw), axis=1)
    var_xi_prev = np.concatenate((var1, var2), axis=0)

    # square root of (N * variance matrix)
    sqrt_var = scipy.linalg.sqrtm(xi_N*var_xi_prev)    

    # generate 2N sigma points
    sigma_xi = np.zeros((xi_N, xi_N*2))
    for i in range(0,xi_N):
        sigma_xi[:,i] = xi_prev.reshape(xi_N,) + sqrt_var[:,i]
    for i in range(xi_N,xi_N*2):
        sigma_xi[:,i] = xi_prev.reshape(xi_N,) - sqrt_var[:,i-xi_N]

    # compute the prior sigma points with dynamics equation
    sigma_xp = q(sigma_xi, steeringAngle, pedalSpeed, B, r, dt)

    # compute prior statistics
    x_est = np.mean(sigma_xp, axis=1)
    Pm = np.cov(sigma_xp)
    
        ## Step 2. measurement update if there are a valid measurement
    if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
        # calculate sigma_z points using measurment equation
        sigma_z = h(sigma_xp, B)

        # compute z_hat, Pzz, Pxz from sigma_z points and get Pzz
        z_est = np.mean(sigma_z, axis=1)
        Pzz = np.cov(sigma_z) + np.eye(2) * 1e-6

        # calculate Pxz
        Pxz = np.zeros((x_N,2))
        for i in range(xi_N*2):
            Pxz += (sigma_xp[:,i] - x_est).reshape(x_N,1) @ (sigma_z[:,i] - z_est).reshape(1,2)
        Pxz = Pxz / (xi_N*2)

        # apply kalman filter
        K = Pxz @ np.linalg.inv(Pzz) # 3 by 2
        z = np.array([[measurement[0]],
                      [measurement[1]]])
        x_est = x_est.reshape(x_N,1) + K@(z - z_est.reshape(2,1)) 
        Pm = Pm - K@Pzz@K.T
    
    # results of UKF run
    x = np.squeeze(x_est[0])
    y = np.squeeze(x_est[1])
    theta = np.squeeze(x_est[2])

    return np.array([x, y, theta]), Pm


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

    ################## MY CODE STARTS ##########################d

    Br_list = [(0.72,0.40375),(0.72,0.44625),
               (0.88,0.40375),(0.88,0.44625),
               (0.8,0.425)]
    #Br_list = [(0.8,0.425)]
    trials = len(Br_list)
    output_state = np.zeros((3,trials)) # 5 variables * 4 implementations
    output_Pm = [None] * trials

    for i, (B, r) in enumerate(Br_list):
        output_state[:,i], output_Pm[i] = implement_UKF(dt, internalStateIn, steeringAngle, pedalSpeed, measurement, B, r)

    ################## MY CODE ENDS ##########################

    #### OUTPUTS ####
    # Update the internal state (will be passed as an argument to the function
    # at next run), must obviously be compatible with the format of internalStateIn:

    internalStateOut = list(np.mean(output_state, axis=1)) # x, y, theta
    internalStateOut.append(np.mean(output_Pm, axis=0)) # append Pm at the end


    x = internalStateOut[0]
    y = internalStateOut[1]
    theta = internalStateOut[2]

    # DO NOT MODIFY THE OUTPUT FORMAT:
    return x, y, theta, internalStateOut 



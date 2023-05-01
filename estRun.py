import numpy as np
import scipy as sp
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

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
    Sw = np.array([[1.0880701, 0.        ],
                   [0.       , 2.98447239]])

    # storing x(k-1)
    x_prev = internalStateIn[0]
    y_prev = internalStateIn[1]
    theta_prev = internalStateIn[2]
    vel_prev = internalStateIn[3]
    gamma_prev = internalStateIn[4]
    Pm_prev = internalStateIn[5]

    # 1. Prior Update (applying dynamics equation)
    x = x_prev + dt*vel_prev*np.cos(theta_prev)
    y = y_prev + dt*vel_prev*np.sin(theta_prev)
    theta = theta_prev + dt*vel_prev*np.tan(gamma_prev)/B
    vel = 5*r*pedalSpeed
    gamma = steeringAngle
    x_est = np.array([x, y, theta, vel, gamma]).reshape(5,1)

    A = [[1, 0, -dt*vel_prev*np.sin(theta_prev), dt*np.cos(theta_prev), 0],
         [0, 1, +dt*vel_prev*np.cos(theta_prev), dt*np.sin(theta_prev), 0],
         [0, 0, 1, dt*np.tan(gamma_prev)/B, dt*vel_prev/B/np.cos(gamma_prev)**2],
         [0,0,0,0,0],
         [0,0,0,0,1]]
    A = np.asarray(A)
    P = A@Pm_prev@A.T

    if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
        # have a valid measurement => Measurement update
        z = np.array([[measurement[0]],
                       [measurement[1]]])
        h_x = np.array([[x+0.5*B*np.cos(theta)],
                        [y+0.5*B*np.sin(theta)]])
        H = np.array([[1, 0, -0.5*B*np.sin(theta_prev), 0, 0],
                      [0, 1, 0.5*B*np.cos(theta_prev), 0, 0]])
        M = np.array([[1, 0], [0, 1]])

        K = P @ H.T @ np.linalg.inv(H@P@H.T + M@Sw@M.T) # 5 by 2 matrix
        x_est = x_est.reshape(5,1) + K@(z - h_x)
        P = (np.eye(5) - K@H)@P
    
        x = x_est[0,0]
        y = x_est[1,0]
        theta = x_est[2,0]
        vel = x_est[3,0]
        gamma = x_est[4,0]

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
                     P]

    # DO NOT MODIFY THE OUTPUT FORMAT:
    return x, y, theta, internalStateOut 



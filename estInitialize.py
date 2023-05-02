import numpy as np
import scipy as sp
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

def estInitialize():
    # Fill in whatever initialization you'd like here. This function generates
    # the internal state of the estimator at time 0. You may do whatever you
    # like here, but you must return something that is in the format as may be
    # used by your estRun() function as the first returned variable.
    #
    # The second returned variable must be a list of student names.
    # 
    # The third return variable must be a string with the estimator type

    #we make the internal state a list, with the first three elements the position
    # x, y; the angle theta; and our favorite color. 
    x = 0.83725424
    y = 0.97107426
    theta = -0.15041826
    vel= 4.06594596
    gamma = -0.07650388
    Pm =   [[ 7.09513132e+00,  1.17695984e+00, -5.14362163e-01,  2.14779854e+00,  9.75766042e-03],
              [ 1.17695984e+00,  1.51932215e+01, -1.28025108e+00,  2.32226751e-01,  1.35511061e-01],
              [-5.14362163e-01, -1.28025108e+00,  2.47015399e+00,  6.21730260e-01,  -2.99331352e-02],
              [ 2.14779854e+00,  2.32226751e-01,  6.21730260e-01,  1.17378205e+01,  1.35709979e-01],
              [ 9.75766042e-03,  1.35511061e-01, -2.99331352e-02,  1.35709979e-01,  1.70354315e-01]]
    Pm = np.asarray(Pm)
    # note that there is *absolutely no prescribed format* for this internal state.
    # You can put in it whatever you like. Probably, you'll want to keep the position
    # and angle, and probably you'll remove the color.
    internalState = [x,
                     y,
                     theta,
                     vel,
                     gamma,
                     Pm]

    # replace these names with yours. Delete the second name if you are working alone.
    studentNames = ['Austin Kim']
    
    # replace this with the estimator type. Use one of the following options:
    #  'EKF' for Extended Kalman Filter
    #  'UKF' for Unscented Kalman Filter
    #  'PF' for Particle Filter
    #  'OTHER: XXX' if you're using something else, in which case please
    #                 replace "XXX" with a (very short) description
    estimatorType = 'EKF'  
    
    return internalState, studentNames, estimatorType


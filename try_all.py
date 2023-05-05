import numpy as np
import matplotlib.pyplot as plt
from estRun import estRun
from estInitialize import estInitialize
import time

trial = 99

final_error_x = [None]*trial
final_error_y = [None]*trial
final_error_theta = [None]*trial
final_error_dist = [None]*trial

start_time = time.time()

for i in range(1,trial+1):
    print('Running #', i)
    experimentalData = np.genfromtxt ('data/run_{0:03d}.csv'.format(i), delimiter=',')
    internalState, studentNames, estimatorType = estInitialize()

    numDataPoints = experimentalData.shape[0]

    #Here we will store the estimated position and orientation, for later plotting:
    estimatedPosition_x = np.zeros([numDataPoints,])
    estimatedPosition_y = np.zeros([numDataPoints,])
    estimatedAngle = np.zeros([numDataPoints,])
    est_vel = np.zeros([numDataPoints,])

    dt = experimentalData[1,0] - experimentalData[0,0]
    for k in range(numDataPoints):
        t = experimentalData[k,0]
        gamma = experimentalData[k,1]
        omega = experimentalData[k,2]
        measx = experimentalData[k,3]
        measy = experimentalData[k,4]
        
        #run the estimator:
        x, y, theta, internalState = estRun(t, dt, internalState, gamma, omega, (measx, measy))

        #keep track:
        estimatedPosition_x[k] = x
        estimatedPosition_y[k] = y
        estimatedAngle[k] = theta
        #est_vel[k] = internalState[3]
    
    #make sure the angle is in [-pi,pi]
    estimatedAngle = np.mod(estimatedAngle+np.pi,2*np.pi)-np.pi

    posErr_x = estimatedPosition_x - experimentalData[:,5]
    posErr_y = estimatedPosition_y - experimentalData[:,6]
    angErr   = np.mod(estimatedAngle - experimentalData[:,7]+np.pi,2*np.pi)-np.pi

    final_error_x[i-1] = posErr_x[-1]
    final_error_y[i-1] = posErr_y[-1]
    final_error_theta[i-1] = angErr[-1]
    final_error_dist[i-1] = np.sqrt(posErr_x[-1]**2 + posErr_y[-1]**2)
end_time = time.time()

print('mean of dist error = ', np.mean(final_error_dist))
print('elapsed time = ', end_time - start_time)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].hist(final_error_x, bins=10)
axs[0, 0].set_title('final_error_x')
axs[0, 1].hist(final_error_y, bins=10)
axs[0, 1].set_title('final_error_y')
axs[1, 0].hist(final_error_theta, bins=10)
axs[1, 0].set_title('final_error_theta')
axs[1, 1].hist(final_error_dist, bins=10)
axs[1, 1].set_title('final_error_dist')
plt.show()
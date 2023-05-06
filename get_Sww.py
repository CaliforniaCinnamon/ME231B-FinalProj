import numpy as np

nparray = np.genfromtxt ('data/run_{0:03d}.csv'.format(0), delimiter=',')
mask = ~np.isnan(nparray[:, 3]) & ~np.isnan(nparray[:, 4])
non_nan_rows = nparray[mask]
meas_data = non_nan_rows[:,3:5]

real_x = nparray[-1,5]
real_y = nparray[-1,6]
real_theta = nparray[-1,7]
B = 0.87617041920118446300878192822554 # optimized B length
mu_w = np.array([real_x+B*np.cos(real_theta),
                 real_y+B*np.cos(real_theta)])


w = meas_data - mu_w
Sww_vec = np.var(w, axis=0)
Sww = np.diag(Sww_vec)

print(np.mean(w, axis=0))

# try mu_w = (0,0)
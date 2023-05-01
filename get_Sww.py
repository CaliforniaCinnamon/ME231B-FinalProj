import numpy as np

nparray = np.genfromtxt ('data/run_{0:03d}.csv'.format(0), delimiter=',')
mask = ~np.isnan(nparray[:, 3]) & ~np.isnan(nparray[:, 4])
non_nan_rows = nparray[mask]
meas_data = non_nan_rows[:,3:5]

real_x = nparray[-1,5]
real_y = nparray[-1,6]
real_theta = nparray[-1,7]
R = 0.87617041920118446300878192822554
mu_w = np.array([real_x+R*np.cos(real_theta),
                 real_y+R*np.cos(real_theta)])

w = meas_data - mu_w
Sww_vec = np.var(w, axis=0)
Sww = np.diag(Sww_vec)

print(np.mean(w, axis=0))
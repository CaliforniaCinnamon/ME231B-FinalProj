import numpy as np


x0_arr = np.zeros((100,5))

for i in range(100):
    nparray = np.genfromtxt ('data/run_{0:03d}.csv'.format(i), delimiter=',')

    # obatain the first available measurement
    mask = ~np.isnan(nparray[:, 3]) & ~np.isnan(nparray[:, 4])
    non_nan_rows = nparray[mask]
    earliest_row_idx = np.argmin(non_nan_rows[:, 0])
    second_earlist_row_idx = np.argsort(non_nan_rows[:, 0])[1]
    x0_arr[i,0] = non_nan_rows[earliest_row_idx, 3]
    x0_arr[i,1] = non_nan_rows[earliest_row_idx, 4]
    zx_2 = non_nan_rows[second_earlist_row_idx, 3]
    zy_2 = non_nan_rows[second_earlist_row_idx, 4]

    # obtain theta, vel, gamma from the first row of data
    x0_arr[i,2] = np.arctan2((zy_2 - x0_arr[i,1]),(zx_2 - x0_arr[i,0]))
    x0_arr[i,3] = 5*(0.425)*nparray[0,2]
    x0_arr[i,4] = nparray[0,1]

x0 = np.mean(x0_arr, axis=0)
#p0 = np.var(x0_arr, axis=0)
#P0 = np.diag(p0)
P0 = np.cov(x0_arr, rowvar=False)

print(x0)
print(P0)

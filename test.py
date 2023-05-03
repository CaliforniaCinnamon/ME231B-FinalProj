import numpy as np

a = np.array([11,654,65])
b = np.cov(a.T, a.T)

print(b)
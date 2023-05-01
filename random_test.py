import numpy as np

# Create sample matrices
mat1 = np.random.rand(5, 5)
mat2 = np.random.rand(1, 1)
mat3 = np.random.rand(2, 2)

# Create diagonal matrices from the input matrices
d1 = np.diag(np.diag(mat1))
d2 = np.diag(np.diag(mat2))
d3 = np.diag(np.diag(mat3))

# Concatenate the diagonal matrices along the appropriate axis
result = np.concatenate((d1, d2, d3), axis=0)

print(mat1)
print(d1)
print(result)
import numpy as np

# Load both matrices (no headers, no indices)
A = np.loadtxt("sparse_matrix_normal.csv", delimiter=",")
B = np.loadtxt("sparse_matrix_normal2.csv", delimiter=",")

# Matrix multiplication
C = A @ B        # same as np.dot(A, B)

# Save the result
np.savetxt("matrix_C.csv", C, delimiter=",")
import numpy as np
import pandas as pd
from scipy.sparse import random as sparse_random

# Create a 40x40 sparse matrix with ~30% nonzero entries
sparse_mat = sparse_random(3000, 10, density=0.3, format='coo')

# Convert to dense 2D numpy array (fills missing values with 0s)
dense_mat = sparse_mat.toarray()

# Convert to DataFrame (optional)
df = pd.DataFrame(dense_mat)

# Save as CSV including all 0s
df.to_csv("sparse_matrix_normal3.csv", index=False, header=False)

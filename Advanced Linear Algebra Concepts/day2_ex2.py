import numpy as np

A = np.array([[4, -2],[1, 1]])

eigvals, eigvec = np.linalg.eig(A)

print("EigenValues: ",eigvals)
print("EigenVectors: \n",eigvec)
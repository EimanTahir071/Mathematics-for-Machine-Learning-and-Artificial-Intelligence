import numpy as np

A = np.array([[2, 3, 4], [4, 5, 6], [7, 8, 9]])

determinant = np.linalg.det(A)
inverse = np.linalg.inv(A)

print("Determinant: ", determinant)
print("Inverse: ", inverse)
import numpy as np

A = np.array([[2, 3, 4], [4, 5, 6], [7, 8, 9]])

determinant = np.linalg.det(A)
print("Determinant: ", determinant)

if abs(determinant) > 1e-10:
    inverse = np.linalg.inv(A)
    print("Inverse: ", inverse)
else:
    print("Matrix is singular; inverse does not exist.")

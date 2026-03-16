import numpy as np

# Create a square matrix
A = np.array([[1, 2, 3],
              [0, 4, 5],
              [1, 3, 6]])

# Transpose
print("Transpose:\n", A.T)

# Determinant
det = np.linalg.det(A)
print("Determinant:", det)

# Inverse (only valid if determinant is non-zero)
if abs(det) > 1e-10:
    inv = np.linalg.inv(A)
    print("Inverse:\n", inv)
    # Verify: A @ A_inv should be identity (rounded to remove floating-point noise)
    print("A @ A_inv:\n", np.round(A @ inv, decimals=10))
else:
    print("Matrix is singular, inverse does not exist.")

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

import numpy as np

A = np.array([[2, 3], [1,4]])
determinant = np.linalg.det(A)
# print("Determinant: ", determinant)


U, S, Vt = np.linalg.svd(A)
print("U: \n", U)
print("Singular Values: \n", S)
print("V Transpose: \n", Vt)



inverse = np.linalg.inv(A)
# print("Inverse of A: \n", inverse)

eigenValues, eigneVectors = np.linalg.eig(A)
# print("EigenVal\n", eigenValues)
# print("EigenVectors\n", eigneVectors)

B = np.array([[4, 2], [1, 1]])
eigval, eigvec = np.linalg.eig(B)
# print("EigVal: ",eigval)
# print("EigVect: \n",eigvec)
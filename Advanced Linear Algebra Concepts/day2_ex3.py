"""
Singular Value Decomposition (SVD) demonstration using NumPy.

SVD decomposes a matrix A into three matrices: A = U @ Sigma @ Vt, where:
  - U  : orthogonal matrix whose columns are the left singular vectors of A
  - S  : 1-D array of singular values (non-negative, sorted in descending order)
  - Vt : orthogonal matrix whose rows are the right singular vectors of A
"""

import numpy as np

A = np.array([[3, 1, 1], [-1, 3, 1], [1, 1, 3]])
print("Original Matrix A:\n", A)

# Compute SVD
U, S, Vt = np.linalg.svd(A)

# U: left singular vectors (columns form an orthonormal basis for the column space of A)
print("\nU (Left Singular Vectors):\n", np.round(U, 4))

# S: singular values — measure the "importance" of each component
print("\nSingular Values:\n", np.round(S, 4))

# Vt: right singular vectors transposed (rows form an orthonormal basis for the row space of A)
print("\nVt (Right Singular Vectors Transposed):\n", np.round(Vt, 4))

# ── Reconstruct A = U @ Sigma @ Vt ──────────────────────────────────────────
# Use np.diag(S) to build the diagonal Sigma matrix concisely
Sigma = np.diag(S)
reconstructed = U @ Sigma @ Vt
print("\nReconstructed Matrix:\n", np.round(reconstructed, 4))

# Verify that reconstruction matches the original matrix
if np.allclose(A, reconstructed):
    print("\nVerification: Reconstruction matches the original matrix. ✓")
else:
    print("\nVerification: Reconstruction does NOT match the original matrix. ✗")

# ── Low-Rank Approximation ───────────────────────────────────────────────────
# Keep only the top-k singular values/vectors to form a rank-k approximation.
# This is useful for dimensionality reduction, image compression, etc.
k = 2  # rank-2 approximation
A_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
print(f"\nRank-{k} Approximation of A:\n", np.round(A_approx, 4))

# The Frobenius-norm error shows how much information is lost
error = np.linalg.norm(A - A_approx, ord="fro")
print(f"Approximation error (Frobenius norm): {error:.4f}")
"""
Day 3 Exercise 2 – Gradient Descent using Derivatives.

Gradient descent is the backbone of ML optimisation: it uses the derivative
(or gradient) to iteratively move toward the minimum of a loss function.

Covers:
  - 1-D gradient descent on a simple quadratic
  - Tracking the loss at every step
"""

import sympy as sp

x = sp.Symbol('x')

# ── Loss function ────────────────────────────────────────────────────────────
# f(x) = (x - 3)²  has a global minimum at x = 3
f = (x - 3)**2
df = sp.diff(f, x)
print("Loss  f(x)  =", f)
print("Grad  f'(x) =", df)

# ── Gradient Descent Loop ────────────────────────────────────────────────────
learning_rate = 0.1
x_val = 0.0          # starting point
n_iterations = 20

print(f"\nStarting gradient descent from x = {x_val}")
print(f"{'Iter':>4}  {'x':>10}  {'f(x)':>10}")
print("-" * 30)

for i in range(1, n_iterations + 1):
    grad = float(df.subs(x, x_val))       # evaluate gradient at current x
    x_val = x_val - learning_rate * grad   # update step
    loss  = float(f.subs(x, x_val))
    print(f"{i:>4}  {x_val:>10.6f}  {loss:>10.6f}")

print(f"\nConverged to x ≈ {x_val:.6f}  (expected 3.0)")

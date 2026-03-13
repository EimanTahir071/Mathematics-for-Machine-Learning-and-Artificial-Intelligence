"""
Day 3 Exercise 3 – Numerical Differentiation and Higher-Order Derivatives.

When an analytic derivative is unavailable, we can approximate it numerically.
Higher-order derivatives (Hessian diagonal) provide curvature information used
in second-order optimisers.

Covers:
  - Central-difference approximation of the first derivative
  - Second derivative (curvature) via central differences
  - Comparing numerical result with the analytic (SymPy) answer
  - Partial derivatives of a multivariate function
"""

import sympy as sp
import numpy as np

# ── Numerical vs Analytic Derivative ────────────────────────────────────────
def numerical_derivative(func, x_val, h=1e-5):
    """Central-difference approximation: [f(x+h) - f(x-h)] / (2h)."""
    return (func(x_val + h) - func(x_val - h)) / (2 * h)

def numerical_second_derivative(func, x_val, h=1e-5):
    """Central-difference approximation: [f(x+h) - 2f(x) + f(x-h)] / h²."""
    return (func(x_val + h) - 2 * func(x_val) + func(x_val - h)) / h**2

# f(x) = x³ – 4x
f_numeric = lambda x: x**3 - 4*x

test_point = 2.0
num_deriv  = numerical_derivative(f_numeric, test_point)
num_deriv2 = numerical_second_derivative(f_numeric, test_point)

# Analytic answer via SymPy
x = sp.Symbol('x')
f_sym = x**3 - 4*x
analytic_deriv  = float(sp.diff(f_sym, x).subs(x, test_point))
analytic_deriv2 = float(sp.diff(f_sym, x, 2).subs(x, test_point))

print(f"Function: f(x) = x³ - 4x   at x = {test_point}")
print(f"  First derivative  — numerical: {num_deriv:.6f},  analytic: {analytic_deriv:.6f}")
print(f"  Second derivative — numerical: {num_deriv2:.6f},  analytic: {analytic_deriv2:.6f}")

# ── Partial Derivatives of a Multivariate Function ──────────────────────────
# g(x, y) = x²y + sin(y)
x, y = sp.symbols('x y')
g = x**2 * y + sp.sin(y)

dg_dx = sp.diff(g, x)   # ∂g/∂x
dg_dy = sp.diff(g, y)   # ∂g/∂y

print("\nMultivariate function: g(x, y) = x²y + sin(y)")
print("  ∂g/∂x =", dg_dx)
print("  ∂g/∂y =", dg_dy)

# Evaluate gradient at (x=1, y=π/2)
point = {x: 1, y: sp.pi / 2}
print(f"\n  Gradient at (1, π/2):")
print(f"    ∂g/∂x = {float(dg_dx.subs(point)):.4f}")
print(f"    ∂g/∂y = {float(dg_dy.subs(point)):.4f}")

"""
Day 3 Exercise 1 – Symbolic Derivatives and the Chain Rule using SymPy.

Covers:
  - Basic single-variable derivatives
  - Product rule and quotient rule
  - Chain rule for composite functions
"""

import sympy as sp

x = sp.Symbol('x')

# ── Basic Derivative ─────────────────────────────────────────────────────────
f = x**3 - 5*x**2 + 6*x - 2
df = sp.diff(f, x)
print("f(x)  =", f)
print("f'(x) =", df)

# ── Product Rule: d/dx [u(x) * v(x)] = u'v + uv' ────────────────────────────
u = x**2
v = sp.sin(x)
product = u * v
d_product = sp.diff(product, x)
print("\nProduct rule: d/dx [x² · sin(x)] =", d_product)

# ── Quotient Rule: d/dx [u/v] = (u'v - uv') / v² ───────────────────────────
numerator = sp.exp(x)
denominator = x**2 + 1
quotient = numerator / denominator
d_quotient = sp.diff(quotient, x)
print("\nQuotient rule: d/dx [e^x / (x²+1)] =", sp.simplify(d_quotient))

# ── Chain Rule: d/dx [f(g(x))] = f'(g(x)) · g'(x) ──────────────────────────
g = x**2 + 1          # inner function
composite = sp.sin(g)  # outer function applied to g
d_composite = sp.diff(composite, x)
print("\nChain rule: d/dx [sin(x²+1)] =", d_composite)

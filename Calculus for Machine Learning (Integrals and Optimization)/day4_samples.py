import sympy as sp

x = sp.Symbol('x')
f = x**2
definite_integral = sp.integrate(f, (x, 0, 2))
indefinite_integral = sp.integrate(f, x)
print("Definite Integral:", definite_integral)
print("Indefinite Integral:", indefinite_integral)
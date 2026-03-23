import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Poisson Distribution
lam = 3
x = np.arange(0, 10)
y = poisson.pmf(x, lam)
plt.bar(x, y, label="Poisson")
plt.title("Poisson Distribution")
plt.show()

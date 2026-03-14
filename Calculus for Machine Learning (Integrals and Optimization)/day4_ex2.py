import numpy as np

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term to X
X_b = np.c_[np.ones((100, 1)), X]

# SGD Implementaion
def stochastic_gradient_descent(X, y, theta, learning_rate, n_epocs):
    m = len(y)
    for epoch in range(n_epocs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T @ (xi @ theta - yi)
            theta -= learning_rate * gradients
    return theta

# initialize parameters
theta = np.random.randn(2, 1)
learning_rate = 0.01
n_epochs = 50

# Perform SGD
theta_opt = stochastic_gradient_descent(X_b, y, theta, learning_rate, n_epochs)
print("Optimized Parameters:", theta_opt)
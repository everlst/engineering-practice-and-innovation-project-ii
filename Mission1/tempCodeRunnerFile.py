import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x_values = np.linspace(-10, 10, 400)
y_values = sigmoid(x_values)

plt.plot(x_values, y_values)
plt.title("Sigmoid Activation Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

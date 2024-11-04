import matplotlib.pyplot as plt
import numpy as np


def func(x):
    return 6 / np.sqrt(x[0] ** 2 + x[1] ** 2) - (
        1 / np.sqrt((x[0] - 1) ** 2 + x[1] ** 2)
    )


x = np.arange(-1.5, 2, 0.1)
y = np.arange(-1.5, 2, 0.1)
X, Y = np.meshgrid(x, y)

figure = plt.figure()
axis = plt.axes(projection="3d")
results = func([X, Y])
axis.plot_surface(X, Y, results, cmap="winter")
plt.show()

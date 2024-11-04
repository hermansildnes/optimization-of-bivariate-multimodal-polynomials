import numpy as np
import matplotlib.pyplot as plt


def animation(start):
    (x, y, z) = start
    print(x, y, z)


def rosenbrock(x, y):
    return 100 * (y - x ** 2) ** 2 + (x - 1) ** 2


def ackley(x, y):
    return (
        -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        + np.e
        + 20
    )


# define range for input
r_min, r_max = -5.0, 5.0
# sample input range uniformly at 0.1 increments
xaxis = np.arange(r_min, r_max, 0.1)
yaxis = np.arange(r_min, r_max, 0.1)
# create a mesh from the axis
x, y = np.meshgrid(xaxis, yaxis)
# compute targets
results = ackley(x, y)
print(type(x))
print(results, type(results))
# create a surface plot with the jet color scheme
figure = plt.figure()
axis = figure.gca(projection="3d")
axis.plot_surface(x, y, results, cmap="winter")
axis.scatter(0, 0, -1)
# show the plot
plt.show()

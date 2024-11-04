import matplotlib.pyplot as plt
import numpy as np


def func(x, y):
    return x**2 + y**2


x, y = np.meshgrid(np.arange(-50, 50), np.arange(-50, 50))
results = func(x, y)

figure = plt.figure()
axis = figure.gca(projection="3d")

point = 10
stepsize = 0.1
pointlist = [point]
for i in range(31):
    gradient = 2 * point
    point = point - stepsize * gradient
    print(point)
    # pointlist.append(point)


axis.plot_surface(x, y, results, cmap="winter")
axis.scatter(point, point, c="r")
# for points in pointlist:
#     axis.scatter(points, points, results[round(points), round(points)], c="r")

plt.show()

import ijson
import matplotlib.pyplot as plt
from tqdm import tqdm
import functions
import numpy as np


def main():
    shape = (200, 200)
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    X, Y = np.meshgrid(x, y)
    with open("coefficients.json", "r") as file:
        parser = ijson.items(file, "functions.item", use_float=True)

        for obj in parser:
            order = obj["order"]
            break

        for obj in tqdm(parser, total=5000):
            general_basis = functions.get_general_basis(obj["coefficients"], order)

            print(
                general_basis.replace("+ -", "- ")
                .replace("x[0]", "x")
                .replace("x[1]", "y")
            )

            plt.figure()
            axis = plt.axes(projection="3d")
            results = functions.evaluatePolynomial([X, Y], general_basis)
            axis.plot_surface(X, Y, results, cmap="winter")

            plt.figure()
            axis2 = plt.axes(projection="3d")
            axis2.plot_surface(X, Y, np.array(obj["perlin_matrix"]), cmap="winter")

            plt.show()


if __name__ == "__main__":
    main()

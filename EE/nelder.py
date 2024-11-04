import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import argParser as ap
import functions
from scipy import optimize

np.random.seed(0)
shape = (200, 200)
x = np.arange(shape[0])
y = np.arange(shape[1])
X, Y = np.meshgrid(x, y)


def main():
    args = ap.parse()

    # Reads the functions from the JSON file
    if args.action == "read":
        import ijson
        import json

        with open("coefficients.json", "r") as file:
            parser = ijson.items(file, "functions.item", use_float=True)

            for obj in parser:
                order = obj["order"]
                break

            data = {"nelder_mead": []}

            cutoff = 5000

            for obj in tqdm(parser, total=cutoff):
                general_basis = functions.get_general_basis(obj["coefficients"], order)

                starttime = time.process_time()

                x_min = optimize.minimize(
                    functions.evaluatePolynomial,
                    x0=[
                        np.random.randint(10, shape[0] - 10),
                        np.random.randint(10, shape[1] - 10),
                    ],
                    args=(general_basis),
                    bounds=[(1, 199), (1, 199)],
                    method="Nelder-Mead",
                    tol=1e-4,
                )

                time_taken = time.process_time() - starttime

                zbest = functions.evaluatePolynomial(x_min.x, general_basis)

                data["nelder_mead"].append(
                    {
                        "index": obj["index"],
                        "xbest": x_min.x[0],
                        "ybest": x_min.x[1],
                        "zbest": zbest,
                        "absolute_minimum": obj["absolute_min"],
                        "error": np.abs(zbest - obj["absolute_min"]),
                        "time_taken": time_taken,
                        "iterations": x_min.nit,
                        "success": x_min.success,
                    }
                )

                # figure = plt.figure()
                # axis = figure.gca(projection="3d")
                # results = functions.evaluatePolynomial([X, Y], general_basis)
                # axis.plot_surface(X, Y, results, cmap="winter")
                # # plot nelder mead minima
                # axis.scatter(x_min.x[0], x_min.x[1], zbest, c="r")
                # # plot gradient descent minima
                # plt.show()

                if obj["index"] == cutoff:
                    break

        with open("nelder_mead.json", "w") as file:
            json.dump(data, file, indent=4)


if __name__ == "__main__":
    main()

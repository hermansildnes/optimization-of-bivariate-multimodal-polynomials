import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import argParser as ap
import functions

np.random.seed(0)
shape = (200, 200)
x = np.arange(shape[0])
y = np.arange(shape[1])
X, Y = np.meshgrid(x, y)


def main():
    args = ap.parse()

    if args.action == "read":
        import ijson
        import json
        import algorithms as algos

        with open("coefficients.json", "r") as file:
            parser = ijson.items(file, "functions.item", use_float=True)

            for obj in parser:
                order = obj["order"]
                break

            data = {"gradient_descent": []}

            cutoff = 5000

            for obj in tqdm(parser, total=cutoff):
                general_basis = functions.get_general_basis(obj["coefficients"], order)
                success = False

                while not success:
                    starttime = time.process_time()

                    zbest, xybest, steps, success = algos.gradient_descent(
                        functions.evaluatePolynomial,
                        [
                            np.random.randint(10, shape[0] - 10),
                            np.random.randint(10, shape[1] - 10),
                        ],
                        general_basis,
                    )

                    time_taken = time.process_time() - starttime

                data["gradient_descent"].append(
                    {
                        "index": obj["index"],
                        "xbest": xybest[0],
                        "ybest": xybest[1],
                        "zbest": zbest,
                        "absolute_minimum": obj["absolute_min"],
                        "error": np.abs(zbest - obj["absolute_min"]),
                        "time_taken": time_taken,
                        "iterations": steps,
                        "success": success,
                    }
                )

                # figure = plt.figure()
                # axis = plt.axes(projection="3d")
                # results = functions.evaluatePolynomial([X, Y], general_basis)
                # axis.plot_surface(X, Y, results, cmap="winter")
                # axis.scatter(xybest[0], xybest[1], zbest, c="r")
                # plt.show()

                if obj["index"] == cutoff:
                    break

        with open("gradient_descent.json", "w") as file:
            json.dump(data, file, indent=4)


if __name__ == "__main__":
    main()

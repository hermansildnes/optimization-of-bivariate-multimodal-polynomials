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

    if args.action == "test":
        import algorithms as algos

    # Generate file with *amount* functions of *order* + coeffs + data about the functions
    if args.action == "write":
        import functionGenerator as fg

        fg.genPerlinCoeffs(args.amount, args.order, "coefficients.json", shape=shape)

    # Reads the functions from the JSON file
    if args.action == "read":
        import ijson
        import json
        import algorithms as algos

        with open("coefficients.json", "r") as file:
            parser = ijson.items(file, "functions.item", use_float=True)
            # Check *order* of the functions in the JSON file
            for obj in parser:
                order = obj["order"]
                break

            data = {"gradient_descent": []}

            successes = 0
            cutoff = 100
            success = False

            # Code applied to all saved functions go in here (e.g. plotting, optimization)
            for obj in tqdm(parser, total=cutoff):
                general_basis = functions.get_general_basis(obj["coefficients"], order)

                # In case of NaN
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

                if success:
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
                    successes += 1

                if successes == cutoff:
                    break

                # Nelder-Mead using scipy
                # x_min = optimize.minimize(
                #     functions.evaluatePolynomial,
                #     x0=[np.random.randint(0, shape[0]), np.random.randint(0, shape[1])],
                #     args=(obj["coefficients"], order),
                #     bounds=[(1, 199), (1, 199)],
                #     method="Nelder-Mead",
                # )

                # fbest, xbest = algos.minimization(
                #     functions.evaluatePolynomial,
                #     [shape[0] / 2, shape[1] / 2],
                #     (obj["coefficients"], order),
                # )
                # print(f"xbest: {xybest}")
                # print(f"fbest: {zbest}")
                # print(f"absolute: {obj['absolute_min']}\n")
                # print(len(steps))

                # figure = plt.figure()
                # axis = figure.gca(projection="3d")
                # axis.plot_surface(X, Y, results, cmap="winter")
                # # plot nelder mead minima
                # # axis.scatter(x_min.x[0], x_min.x[1], c="r")
                # # plot gradient descent minima

                # axis.scatter(xybest[0], xybest[1], zbest, c="r")
                # plt.show()

        # with open("new.json", "w") as file:
        #     json.dump(data, file, indent=4)

        # with open("anomalies.json", "w") as file:
        #     json.dump(anomalies, file, indent=4)


if __name__ == "__main__":
    main()

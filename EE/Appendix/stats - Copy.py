import numpy as np
import matplotlib.pyplot as plt
import ijson
import statsmodels.api as sm


def main():
    with open("gradient_descent.json", "r") as file:
        parser = ijson.items(file, "gradient_descent.item", use_float=True)
        gerrors = []
        gtimes = []
        gsteps = []
        gsuccesses = 0
        nsuccesses = 0

        for obj in parser:
            if obj["success"] and obj["error"] < 1:
                gerrors.append(obj["error"])
                gtimes.append(obj["time_taken"])
                gsteps.append(obj["iterations"])
                gsuccesses += 1

    print("\nGradient Descent:")
    print(f"successes: {gsuccesses}")
    print(f"error: {sum(gerrors) / gsuccesses}")
    print(f"time: {sum(gtimes) / gsuccesses}")
    print(f"steps: {sum(gsteps) / gsuccesses}")

    with open("nelder_mead.json", "r") as file:
        parser = ijson.items(file, "nelder_mead.item", use_float=True)
        nerrors = []
        ntimes = []
        nsteps = []

        for obj in parser:
            if obj["success"] and obj["error"] < 1:
                nerrors.append(obj["error"])
                ntimes.append(obj["time_taken"])
                nsteps.append(obj["iterations"])
                nsuccesses += 1

    print("\nNelder-Mead:")
    print(f"successes: {nsuccesses}")
    print(f"error: {sum(nerrors) / nsuccesses}")
    print(f"time: {sum(ntimes) / nsuccesses}")
    print(f"steps: {sum(nsteps) / nsuccesses}")

    # sm.qqplot(np.array(gsteps), line="45")

    # sm.qqplot(np.array(nsteps), line="45")

    g_times = plt.figure()
    plt.title("Gradient Descent - Time taken")
    plt.hist(gtimes, bins=np.arange(0, 10, 0.2))

    n_times = plt.figure()
    plt.title("Nelder-Mead - Time taken")
    plt.hist(ntimes, bins=np.arange(0, 1, 0.015))

    g_steps = plt.figure()
    plt.title("Gradient Descent - Iterations")
    plt.hist(gsteps, bins=np.arange(0, 80))

    n_steps = plt.figure()
    plt.title("Nelder-Mead - Iterations")
    plt.hist(nsteps, bins=np.arange(0, 100))

    g_errors = plt.figure()
    plt.title("Gradient Descent - Error")
    plt.hist(gerrors, bins=np.arange(0, 0.5, 0.005))

    n_errors = plt.figure()
    plt.title("Nelder-Mead - Error")
    plt.hist(nerrors, bins=np.arange(0, 0.5, 0.005))

    plt.show()


if __name__ == "__main__":
    main()


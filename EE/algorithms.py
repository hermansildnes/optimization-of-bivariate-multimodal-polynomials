import numpy as np
import numdifftools as nd
from tqdm import tqdm
from scipy.misc import derivative

# Implementation of nelder-mead algorithm using a simplex of size n+1 to find the minimum of function f
def nelder_mead(
    f,
    x_start,
    args,
    step=0.1,
    no_improve_thr=10e-4,
    no_improv_break=10,
    max_iter=1000,
    alpha=1.0,
    gamma=2.0,
    rho=-0.5,
    sigma=0.5,
):
    """
    @param f (function): function to optimize, must return a scalar score
        and operate over a numpy array of the same dimensions as x_start
    @param x_start (numpy array): initial position
    @param step (float): look-around radius in initial step
    @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
        an improvement lower than no_improv_thr
    @max_iter (int): always break after this number of iterations.
        Set it to 0 to loop indefinitely.
    @alpha, gamma, rho, sigma (floats): parameters of the algorithm
        (see Wikipedia page for reference)
    return: tuple (best parameter array, best score)
    """

    # init
    dim = len(x_start)
    prev_best = f(x_start, args)
    no_improv = 0
    res = [[x_start, prev_best]]

    for i in range(dim):
        x = x_start
        x[i] = x[i] + step
        score = f(x, args)
        res.append([x, score])

    # simplex iter
    iters = 0
    while True:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        # break after max_iter
        if max_iter and iters >= max_iter:
            print("\nmaxiter!!\n")
            return res[0][1], res[0][0], max_iter, True
        iters += 1

        # break after no_improv_break iterations with no improvement
        print("...best so far:", best)

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            print("\nnoimprov!!\n")
            return res[0][1], res[0][0], iters, True

        # centroid
        x0 = [0.0] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res) - 1)

        # reflection
        print(x0, alpha, res[-1][0])
        xr = np.array(x0) + alpha * (np.array(x0) - np.array(res)[-1][0])
        rscore = f(xr, args)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = np.array(x0) + gamma * (np.array(x0) - np.array(res)[-1][0])
            escore = f(xe, args)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = np.array(x0) + rho * (np.array(x0) - np.array(res)[-1][0])
        cscore = f(xc, args)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = np.array(x1) + sigma * (np.array(tup[0]) - x1)
            score = f(redx, args)
            nres.append([redx, score])
        res = nres


# Gradient Descent Function
# Here iterations, learning_rate, stopping_threshold
# are hyperparameters that can be tuned
def gradient_descent(
    f,
    x0,
    args,
    iterations=200,
    learning_rate=250,
    stopping_threshold=1e-4,
    momentum=0.8,
):
    x = x0
    fx = f(x, args)

    delta = np.zeros(np.array(x).shape)
    grad = nd.Gradient(f)
    for i in range(iterations):
        try:
            delta = -learning_rate * grad(x, args) + momentum * delta
            x_new = x + delta
        except Exception:
            print(f"\n it: {i}, x: {x}, x0:{x0}\n")
            return fx, x, i, False

        fx_new = f(x_new, args)
        if np.abs(fx_new - fx) < stopping_threshold and i < 5:
            return fx, x, i, False
        if np.abs(fx_new - fx) < stopping_threshold:
            return fx, x, i, True
        x = x_new
        fx = fx_new

    return fx, x, iterations, True

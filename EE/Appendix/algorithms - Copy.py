import numpy as np
import numdifftools as nd
from tqdm import tqdm


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
):
    x = x0
    fx = f(x, args)

    grad = nd.Gradient(f)
    for i in range(iterations):
        try:
            x_new = x - learning_rate * grad(x, args)
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

import numpy as np
import noise
from tqdm import tqdm


def perlin(shape, scale, octaves, persistence, lacunarity, seed):
    z = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            z[i][j] = noise.pnoise2(
                i / scale,
                j / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=seed,
            )
    return z


def get_general_basis(coeffs, order):
    general = ""
    k = 0
    for i in np.arange(0, order + 1, 0.5):
        for j in np.arange(0, order - i + 1, 0.5):
            general += str(coeffs[k]) + "*x[0]**" + str(j) + "*x[1]**" + str(i) + " + "
            k += 1
    return general[:-3]


def get_basis(x: list, y: list, order: int):
    basis = []
    for i in tqdm(np.arange(0, order + 1, 0.5), desc="Creating basis matrix"):
        for j in np.arange(0, order - i + 1, 0.5):
            basis.append(x**j * y**i)
    return basis


def evaluatePolynomial(x, function):
    return eval(function)


def polyfit2d(x, y, z, basis):
    x, y = x.ravel(), y.ravel()
    A = np.vstack(basis).T
    b = z.ravel()
    return np.linalg.lstsq(A, b, rcond=None)





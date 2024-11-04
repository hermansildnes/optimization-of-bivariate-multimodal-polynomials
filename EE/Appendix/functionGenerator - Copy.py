import numpy as np
from tqdm import tqdm
import functions as func
import json


def genPerlinCoeffs(
    runs: int,
    order: int,
    filename: str,
    shape=(200, 200),
    scale=100,
    octaves=6,
    persistence=0.5,
    lacunarity=2,
    seed=325,
):
    data = {"functions": []}
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    X, Y = np.meshgrid(x, y)

    basis = func.get_basis(X.ravel(), Y.ravel(), order)

    for i in tqdm(range(runs), desc="Generating functions"):
        random_scale = scale + np.random.randint(-10, 10)
        random_octaves = octaves + np.random.randint(-1, 1)
        random_persistence = persistence + np.random.uniform(-0.1, 0.1)
        random_lacunarity = lacunarity + np.random.uniform(-0.5, 0.5)
        random_seed = seed + np.random.randint(-325, 325)

        z = func.perlin(
            shape,
            random_scale,
            random_octaves,
            random_persistence,
            random_lacunarity,
            random_seed,
        )

        coeffs, r, rank, s = func.polyfit2d(X, Y, z, basis)
        data["functions"].append(
            {
                "index": i,
                "order": order,
                "scale": random_scale,
                "octaves": random_octaves,
                "lacunarity": random_lacunarity,
                "seed": random_seed,
                "absolute_min": np.min(z),
                "coefficients": list(coeffs),
                "perlin_matrix": z.tolist(),
            }
        )

    with open(filename, "w") as file:
        json.dump(data, file, indent=4)




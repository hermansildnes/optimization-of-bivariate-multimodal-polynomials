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
    # Define the dictionary for saving coefficients + data
    data = {"functions": []}
    # Meshgrid needed for function
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    X, Y = np.meshgrid(x, y)

    # Create the basis function vector
    basis = func.get_basis(X.ravel(), Y.ravel(), order)

    # Generate coefficients from noise *runs* times
    for i in tqdm(range(runs), desc="Generating functions"):
        # Add noise to function-variables
        random_scale = scale + np.random.randint(-10, 10)
        random_octaves = octaves + np.random.randint(-1, 1)
        random_persistence = persistence + np.random.uniform(-0.1, 0.1)
        random_lacunarity = lacunarity + np.random.uniform(-0.5, 0.5)
        random_seed = seed + np.random.randint(-325, 325)

        # Generating the noise
        z = func.perlin(
            shape,
            random_scale,
            random_octaves,
            random_persistence,
            random_lacunarity,
            random_seed,
        )
        # Plotting the Perlin Noise
        # noisefig = plt.figure()
        # noiseaxis = noisefig.gca(projection="3d")
        # noiseaxis.plot_surface(X, Y, z, cmap="viridis")
        # plt.show()

        # Fitting a function to the noise
        coeffs, r, rank, s = func.polyfit2d(X, Y, z, basis)
        # Writing the function to a dictionary for saving to JSON file
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

    # Dump all coefficients to datafile
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

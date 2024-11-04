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
            if obj["index"] == 3:
                general_basis = functions.get_general_basis(obj["coefficients"], order)
                function = (
                    general_basis.replace("+ -", "- ")
                    .replace("x[0]", "x")
                    .replace("x[1]", "y")
                )
                matrix = str(obj["perlin_matrix"]).replace("\n", ", ")
                with open("data.txt", "w") as file2:
                    file2.write(function)
                    file2.write("\n")
                    file2.write(matrix)

            if obj["index"] == 5:
                break


if __name__ == "__main__":
    main()

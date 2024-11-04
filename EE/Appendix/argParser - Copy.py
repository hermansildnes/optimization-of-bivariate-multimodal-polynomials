import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        help="Whether to create new functions or use already existing ones",
        type=str,
    )
    parser.add_argument(
        "-a",
        "--amount",
        help="Amount of functions to generate or perform another action on",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-o",
        "--order",
        help="Order of function to use for regression of noise",
        type=int,
        default=15,
    )
    return parser.parse_args()



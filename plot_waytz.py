#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

KDE_BW = 0.25


def hyp(*args: np.ndarray) -> float:
    """Calculates the Euclidean norm (hypotenuse) of the input vectors.

    Args:
        *args: Variable number of numeric arrays or values to be squared and summed.

    Returns:
        float: The square root of the sum of squares of the input values.
    """
    return np.sqrt(sum([e**2 for e in args]))


def main() -> None:
    """Processes moral concern data and generates visualization plots.

    Reads data from waytz_data.csv, normalizes spatial coordinates, and creates:
    1. A scatter plot showing the distribution of moral concern responses
    2. KDE plots comparing conservative vs liberal moral concern distributions
    """
    d = pd.read_csv("./waytz_data.csv")
    d["x"] = (d["x"] - 360) / (200 * np.sqrt(2))
    d["y"] = -(d["y"] - 270) / (200 * np.sqrt(2))
    plt.scatter(d["x"], d["y"], alpha=0.5)
    plt.xlim(-0.2, 1.0)
    plt.ylim(-0.2, 1.0)
    plt.gca().set_aspect("equal")
    plt.show()

    dd = d[d["ideo"] > 4]  # Conservative condition
    sns.kdeplot(hyp(dd["x"], dd["y"]), label=f"Conservative", bw=KDE_BW)
    dd = d[d["ideo"] < 4]  # Liberal condition
    sns.kdeplot(hyp(dd["x"], dd["y"]), label=f"Liberal", bw=KDE_BW)

    plt.legend(frameon=False)
    plt.xlim(0.0, 1.0)
    plt.xlabel("Sphere of moral concern extents (approximate)")
    plt.ylabel("Observation density")
    plt.show()


if __name__ == "__main__":
    main()

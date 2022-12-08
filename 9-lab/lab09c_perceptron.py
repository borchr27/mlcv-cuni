
import argparse
import numpy as np
import sklearn.linear_model
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import lab09_help


parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--seed", default=None, type=int, help="Seed for RNG.")
parser.add_argument("--data", default="generated", type=str, help="Type of evaluated data: 'debug' as in the previous exercises or 'generated'.")
parser.add_argument("--points", default=20, type=int, help="Number of points per class.")


def debugData() -> tuple[np.ndarray, np.ndarray]:
    """Data from the lecture for debugging."""
    return np.asarray([[3, 3], [-1, -2], [-3, 1]]), np.asarray([1, 1, -1])

def main(args : argparse.Namespace):
    # NOTE: Experiment with scikit-learn perceptron.
    
    if args.data == "generated":
        # Generate random data.
        generator = np.random.RandomState(args.seed)
        shift = [[0, 0], [5, 5]]
        data, labels = lab09_help.generatePerceptronData(generator, args.points, shift, [0, 1])
    elif args.data == "debug":
        # TODO: Use scikit-learn's Perceptron to find hyperplane for our original linear
        # classification problem given by the following data.
        # Compare with the results of your perceptron function.
        # - It was in 'lab07b'.
        data, labels = debugData()
    else:
        raise ValueError("Unknown data type: '{}'.".format(args.data))

    # Train the perceptron.
    perceptron = sklearn.linear_model.Perceptron()
    perceptron.fit(data, labels)

    # Create a meshgrid for decision boundary plotting.
    xDots = np.linspace(np.min(data[:, 0]) - 0.5, np.max(data[:, 0]) + 0.5, 500)
    yDots = np.linspace(np.min(data[:, 1]) - 0.5, np.max(data[:, 1]) + 0.5, 500)
    xx, yy = np.meshgrid(xDots, yDots)
    meshData = np.vstack([xx.ravel(), yy.ravel()]).T

    z = perceptron.predict(meshData)

    colours = np.asarray(["red", "blue"])
    _, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={'aspect': 'equal'})
    ax.pcolormesh(xx, yy, z.reshape(xx.shape), cmap=ListedColormap(colours),  alpha=0.25, shading="auto")
    ax.scatter(data[:, 0], data[:, 1], c=colours[labels])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

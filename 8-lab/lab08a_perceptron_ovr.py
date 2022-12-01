
import argparse
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import lab08_help


parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--seed", default=None, type=int, help="Seed for data generation.")
parser.add_argument("--points", default=50, type=int, help="Number of points generated for each class.")
parser.add_argument("--show_generated", default=False, action="store_true", help="Plots the generated data before computation..")
parser.add_argument("--max_iter", default=None, type=int, help="Maximum iterations of perceptron algorithm.")


def perceptron(z : np.ndarray, u_init : np.ndarray, args : argparse.Namespace) -> np.ndarray:
    """Runs the perceptron algorithm."""
    # TODO: Implement the perceptron algorithm (copy from lab07b).
    # - You can remove args.max_iter stopping criterion because the generated data are linearly separable.
    u = None
    converged = None

    return u, converged

def main(args : argparse.Namespace):
    # NOTE: Multi-class perceptron exercise.

    # For reproducibility
    generator = np.random.RandomState(args.seed)

    # Data generation
    positions = np.asarray([[0, 0], [5, 5], [0, 5], [5, 0]])
    classIds = [0, 1, 2, 3]
    data, labels = lab08_help.kPerceptronData(generator, args.points, positions, 2, classIds)

    colours = np.asarray(['red', 'blue', 'green', 'cyan'])
    if args.show_generated:
        _, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={'aspect': 'equal'})
        ax.scatter(data[:, 0], data[:, 1], marker='o', c=colours[labels - 1])
        plt.tight_layout()
        plt.show()

    # TODO: Train perceptron for every "one vs the rest" classification problem.
    # - You have to temporarily change the labels so that they reflect the "one vs the rest" behaviour (1, -1).
    # - The data is linearly separable so you do not have to use args.max_iter (or set it to sufficiently
    #   high number ~100000 for 50 points).
    # - For each classifier store 'u'.

    
    # Create meshgrid for decision boundary visualisation.
    xDots = np.linspace(np.min(data[:, 0]) - 0.5, np.max(data[:, 0]) + 0.5, 50)
    yDots = np.linspace(np.min(data[:, 1]) - 0.5, np.max(data[:, 1]) + 0.5, 50)
    xx, yy = np.meshgrid(xDots, yDots)
    meshData = np.vstack([np.ones([xx.size]), xx.ravel(), yy.ravel()]).T

    # TODO: Predict labels of meshgrid and visualise the decision boundaries as in previous practicals.
    # - Compute predictions for each linear classifier and select one of the positive ones.
    #   - This is usually realised by np.argmax, even though there is no voting involved.
    predictions = None

    # TODO: The decision boundaries will probably look a little strange (one of the classes will include the
    # centre part of the decision space). Why did this happen?
    _, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={'aspect': 'equal'})
    ax.scatter(data[labels==classIds[0], 0], data[labels==classIds[0], 1], marker='o', color=colours[0])
    ax.scatter(data[labels==classIds[1], 0], data[labels==classIds[1], 1], marker='o', color=colours[1])
    ax.scatter(data[labels==classIds[2], 0], data[labels==classIds[2], 1], marker='o', color=colours[2])
    ax.scatter(data[labels==classIds[3], 0], data[labels==classIds[3], 1], marker='o', color=colours[3])
    ax.pcolormesh(xx, yy, predictions.reshape(xx.shape), cmap=ListedColormap(colours), alpha=0.25, shading="auto")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

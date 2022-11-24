
import argparse
import numpy as np
from typing import Tuple
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import lab07_help

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--data", default="debug", type=str, help="Data type: 'debug' for confirmation of lecture results and 'generated' for proper testing.")
parser.add_argument("--seed", default=None, type=int, help="Seed for data generation.")
parser.add_argument("--points", default=50, type=int, help="Number of points generated for each class.")
parser.add_argument("--max_iter", default=10000, type=int, help="Maximum iterations of perceptron algorithm.")


def debugData() -> Tuple[np.ndarray, np.ndarray]:
    """Data from the lecture for debugging."""
    return np.asarray([[3, 3], [-1, -2], [-3, 1]]), np.asarray([1, 1, -1])

def perceptron(z : np.ndarray, u_init : np.ndarray, args : argparse.Namespace) -> Tuple[np.ndarray, bool]:
    """Runs the perceptron algorithm."""
    # TODO: Implement the body of this function.
    # - z: the data array with shape (N, 3) if you haven't changed it.
    # - u_init: initial parameters of the separator, shape (3,) if you haven't changed it.
    # The function should return the final computed parameters 'u' and whther it has converged
    # withing 'args.max_iter' iterations.
    # - Be careful about multiplying vectors and matrices, specifically, transposing
    #   a 1D vector has no effect so you have to reshape it, e.g., instead of u.T, 
    #   you should use np.reshape(u, [-1, 1]) or np.c_[u].
    # - Further, formulae on slides assume column-major notation, the code is written in
    #   row-major notation.
    u = None
    converged = None

    return u, converged

def main(args : argparse.Namespace):
    if args.data == "debug":
        # TODO: Verify that your algorithm works on debugging data first.
        data, labels = debugData()
    elif args.data == "generated":
        generator = np.random.RandomState(args.seed)
        # Data generation should not be changed until you have finished the assignment and you want to
        # see the results on different distributions.
        # - Specifically, 'labelMultipliers=[1, -1]' should not be changed because it will break the algorithm.
        mu1 = [1, 1]
        sigma1 = [[1, -1], [-1, 2]]
        mu2 = [3, 3]
        sigma2 = [[1, 0], [0, 1]]
        data, labels = lab07_help.generateData(generator, args.points, [mu1, mu2], [sigma1, sigma2], [1, -1])
        
    # TODO: Transform data such that the problem becomes u'*z.
    # - Bias is included in the scalar product.
    z = None

    # TODO: Implement the 'perceptron' function which implements the linear classifier
    # and call it to compute the classification boundary.
    # - change u_init to try different starting parameters.
    u_init = [1, 1, 1]
    u_init = np.asarray(u_init, float)
    u_star, converged = perceptron(z, u_init, args)

    if not converged:
        print("Algorithm couldn't converge in {} iterations.".format(args.max_iter))

    print("Computed parameters: {}".format(u_star))

    # Create data for visualisation.
    rDots = np.linspace(np.min(data[:, 0]) - 0.5, np.max(data[:, 0]) + 0.5, 50)
    cDots = np.linspace(np.min(data[:, 1]) - 0.5, np.max(data[:, 1]) + 0.5, 50)
    rr, cc = np.meshgrid(rDots, cDots)
    meshData = np.vstack([np.ones([rr.size]), rr.ravel(), cc.ravel()]).T

    # Predict labels for data grid.
    predictions = meshData @ np.reshape(u_star, [-1, 1]) >= 0

    # Visualise the decision boundaries.
    _, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={'aspect': 'equal'})
    ax.scatter(data[labels==-1, 0], data[labels==-1 ,1], edgecolors='r', marker='o', facecolors='none', s=20, linewidths=1.5)
    ax.scatter(data[labels==1, 0], data[labels==1 ,1], edgecolors='b', marker='s', facecolors='none', s=20, linewidths=1.5)
    ax.pcolormesh(rr, cc, predictions.reshape(rr.shape), cmap=ListedColormap(['r', 'b']), alpha=0.25, shading="auto")
    ax.set_title("{}".format("Converged" if converged else "Not converged"))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

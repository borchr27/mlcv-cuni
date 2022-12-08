
import argparse
import numpy as np
import sklearn.linear_model
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sklearn.neural_network
import lab09_help


parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--seed", default=None, type=int, help="Seed for RNG.")
parser.add_argument("--points", default=20, type=int, help="Number of points per class.")
parser.add_argument("--draw_generated", default=False, action="store_true", help="Draws the generated points.")
parser.add_argument("--hidden_layer_sizes", nargs="*", default=[10, 5], type=int, help="Size of the MLP hidden layer.")
parser.add_argument("--activation", default="tanh", type=str, help="Activation function for the network.")
parser.add_argument("--max_iter", default=1000, type=int, help="Maximum number iterations of the MLP.")


def main(args : argparse.Namespace):
    # NOTE: XOR problem with noise.
    
    generator = np.random.RandomState(args.seed)
    data, labels = lab09_help.noisyXorData(generator, args.points, 0.25, [0, 1])

    colours = np.asarray(["red", "blue"])
    if args.draw_generated:
        _, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={'aspect': 'equal'})
        ax.scatter(data[:, 0], data[:, 1], c=colours[labels])
        plt.tight_layout()
        plt.show()

    # TODO: Change the topology of the network and activation function and see how
    # the error (FP/FN) changes.
    # - args.hidden_layer_sizes is a list, it can be used on commandline by writing, for instance: --hidden_layer_sizes 10 5
    # - Try creating MLP which will be able to consistently separate XOR data.
    mlp = sklearn.neural_network.MLPClassifier(args.hidden_layer_sizes, activation=args.activation, max_iter=args.max_iter)
    mlp.fit(data, labels)

    # Create a meshgrid for decision boundary plotting.
    xDots = np.linspace(np.min(data[:, 0]) - 0.5, np.max(data[:, 0]) + 0.5, 500)
    yDots = np.linspace(np.min(data[:, 1]) - 0.5, np.max(data[:, 1]) + 0.5, 500)
    xx, yy = np.meshgrid(xDots, yDots)
    meshData = np.vstack([xx.ravel(), yy.ravel()]).T

    predictions = mlp.predict(meshData)

    # TODO: Check the confusion matrix: look at the number of true/false positives/negatives.
    # - Run the training a few times and watch FP/FN, error rate.

    colours = np.asarray(["red", "blue"])
    _, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={'aspect': 'equal'})
    ax.pcolormesh(xx, yy, predictions.reshape(xx.shape), cmap=ListedColormap(colours),  alpha=0.25, shading="auto")
    ax.scatter(data[:, 0], data[:, 1], c=colours[labels])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

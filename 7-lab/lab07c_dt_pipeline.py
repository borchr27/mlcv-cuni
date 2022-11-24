
import argparse
import itertools
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import sklearn.tree
import sklearn.model_selection
import lab07_help

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--task", default="boundary", type=str, help="Executed task: 'boundary' for DT boundary visualisation or 'crossval' for best tree selection.")
parser.add_argument("--seed", default=None, type=int, help="Seed for data generation.")
parser.add_argument("--points", default=50, type=int, help="Number of points generated for each class.")
parser.add_argument("--max_depth", default=None, type=int, help="The maximum depth of the trained trees.")
parser.add_argument("--min_samples_split", default=2, type=int, help="The minimum number of samples required fro a node split.")
parser.add_argument("--min_samples_leaf", default=1, type=int, help="The minimum number of samples, which has to be present in a leaf.")
parser.add_argument("--max_leaf_nodes", default=None, type=int, help="The maximum number of leaf nodes allowed in the tree.")
parser.add_argument("--weight_0", default=1.0, type=float, help="Weight of the class 0 in weighted showcase.")
parser.add_argument("--weight_1", default=100.0, type=float, help="Weight of the class 1 in weighted showcase.")
parser.add_argument("--kfold_split", default=5, type=int, help="Number of folds KFold split in cross-validation.")


def visualizeBoundary(args : argparse.Namespace):
    generator = np.random.RandomState(args.seed)
    # Let's generate the data.
    mu1 = [1, 1]
    sigma1 = [[1, -1], [-1, 2]]
    mu2 = [3, 3]
    sigma2 = [[1, 0], [0, 1]]
    data, labels = lab07_help.generateData(generator, args.points, [mu1, mu2], [sigma1, sigma2], [0, 1])

    # TODO: Fit decision tree classifier and play with the parameters.
    tree = None
    
    # TODO: Create data on a grid for boundary visualisation (like lab07b).
    rr, cc = None, None
    meshData = None

    # TODO: Predict labels for the data grid.
    predictions = None

    # TODO: Visaulise the data.
    # - Draw the boundaries using 'Axes.pcolormesh'.
    # - Plot the tree if you want.
    # - Show the initial data points using 'Axes.scatter'
    _, ax = plt.subplots(1, 2, figsize=(14, 8), subplot_kw={'aspect': 'equal'})
    sklearn.tree.plot_tree(tree, ax=ax[0])
    ax[1].scatter(data[labels==0, 0], data[labels==0 ,1], edgecolors='r', marker='o', facecolors='none', s=20, linewidths=1.5)
    ax[1].scatter(data[labels==1, 0], data[labels==1 ,1], edgecolors='b', marker='s', facecolors='none', s=20, linewidths=1.5)
    ax[1].pcolormesh(rr, cc, predictions.reshape(rr.shape), cmap=ListedColormap(['r', 'b']), alpha=0.25, shading="auto")
    plt.tight_layout()
    plt.show()

def bestTreeSelection(args : argparse.Namespace):
    generator = np.random.RandomState(args.seed)
    # Let's generate the data.
    mu1 = [1, 1]
    sigma1 = [[1, -1], [-1, 2]]
    mu2 = [3, 3]
    sigma2 = [[1, 0], [0, 1]]
    data, labels = lab07_help.generateData(generator, args.points, [mu1, mu2], [sigma1, sigma2], [0, 1])

    # TODO: Split data and labels into training and validation sets. Then, do cross-validation
    # to select the best model parameters - create sets of parameters for decision tree, compute
    # cross-validation error for each and select the parameters with lowest error.
    # NOTE: This is, effectively, a manual implementation of 'sklearn.model_selection.GridSearchCV'
    # - These tools are used to choose the best model and its parameters for a given dataset.

    # TODO: Train decision tree model with the best parameters on all 'data'. Next, generate test data,
    # predict labels for test data and compare the validation and test errors.

    pass

def main(args : argparse.Namespace):
    tasks = {
        "boundary" : visualizeBoundary,
        "crossval" : bestTreeSelection,
    }
    if args.task not in tasks.keys():
        raise ValueError("Unrecognised task: '{}'!".format(args.task))
    tasks[args.task](args)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

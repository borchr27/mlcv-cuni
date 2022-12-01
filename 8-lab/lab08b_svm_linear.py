
import argparse
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import sklearn.svm
import lab08_help


parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--seed", default=None, type=int, help="Seed for data generation.")
parser.add_argument("--points", default=50, type=int, help="Number of points generated for each class.")
parser.add_argument("--show_generated", default=False, action="store_true", help="Plots the generated data before computation.")
parser.add_argument("--C", default=1.0, type=float, help="Soft margin constraint for SVM.")
parser.add_argument("--max_iter", default=1000, type=int, help="Maximum iterations of perceptron algorithm.")


def main(args : argparse.Namespace):
    # TODO: Experiment with parameters of the linear SVM and the generated data.
    # - The code for classification and visualisation is implemented.
    # - Try changing data distribution and parameter C - soft margin penalty.
    # - If you get a warning saying that the algorithm hasn't converged then increase 'args.max_iter'

    # Linear SVM
    generator = np.random.RandomState(args.seed)

    # Data generation.
    mu1 = [1, 1]
    sigma1 = [[0.5, 0], [0, 0.25]]
    mu2 = [4, 2]
    sigma2 = [[0.5, 0], [0, 1]]
    data, labels = lab08_help.mvnGenerateData(generator, args.points, [mu1, mu2], [sigma1, sigma2], [0, 1])

    colours = np.asarray(['red', 'blue'])
    if args.show_generated:
        _, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={'aspect': 'equal'})
        ax.scatter(data[:, 0], data[:, 1], marker='o', c=colours[labels])
        plt.tight_layout()
        plt.show()

    # Train soft margin SVM model.
    linearSvm = sklearn.svm.LinearSVC(C=args.C, max_iter=args.max_iter)
    linearSvm.fit(data, labels)

    # Predict the classes for meshgrid.
    xDots = np.linspace(np.min(data[:, 0]) - 0.5, np.max(data[:, 0]) + 0.5, 150)
    yDots = np.linspace(np.min(data[:, 1]) - 0.5, np.max(data[:, 1]) + 0.5, 150)
    xx, yy = np.meshgrid(xDots, yDots)
    meshData = np.vstack([xx.ravel(), yy.ravel()]).T

    predictions = linearSvm.predict(meshData)

    # Visualise the decision boundaries.
    _, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={'aspect': 'equal'})
    ax.pcolormesh(xx, yy, predictions.reshape(xx.shape), cmap=ListedColormap(colours), alpha=0.25, shading="auto")
    ax.scatter(data[:, 0], data[:, 1], marker='o', c=colours[labels])

    decGrid = linearSvm.decision_function(meshData)
    ax.contour(xx, yy, decGrid.reshape(xx.shape), colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    decision_function = linearSvm.decision_function(data)
    support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]

    ax.scatter(data[support_vector_indices, 0], data[support_vector_indices ,1], edgecolors='k', marker='o', facecolors='none', s=80, linewidths=0.5)
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

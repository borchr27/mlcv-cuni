
import argparse
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import sklearn.svm
import sklearn.model_selection
import lab08_help


parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--task", default="visualize", type=str, help="Executed task: 'visualize' for decision boundary visualisation or 'crossval' for kernel cross-validation.")
parser.add_argument("--seed", default=None, type=int, help="Seed for data generation.")
parser.add_argument("--points", default=100, type=int, help="Number of points generated for each class.")
parser.add_argument("--show_generated", default=False, action="store_true", help="Plots the generated data before computation.")
parser.add_argument("--C", default=1.0, type=float, help="Soft margin constraint for SVM.")
parser.add_argument("--kernel", default="rbf", type=str, help="SVM kernel.")
parser.add_argument("--degree", default=3, type=int, help="Degree of polynomial kernel.")
parser.add_argument("--gamma", default="scale", help="Regularisation of sample influence.")
parser.add_argument("--data", default="circle", type=str, help="Type of data for training.")
parser.add_argument("--kfold_split", default=5, type=int, help="Number of folds in the crossvalidation.")


def mvnData(generator : np.random.RandomState, args : argparse.Namespace):
    mu1 = [1, 1]
    sigma1 = [[0.5, 0], [0, 0.25]]
    mu2 = [6, 2]
    sigma2 = [[0.5, 0], [0, 1]]
    data, labels = lab08_help.mvnGenerateData(generator, args.points, [mu1, mu2], [sigma1, sigma2], [0, 1])
    return data, labels

def circleData(generator : np.random.RandomState, args : argparse.Namespace):
    r1 = np.sqrt(generator.rand(args.points))
    t1 = 2 * np.pi * generator.rand(args.points)
    data1 = np.vstack([r1 * np.cos(t1), r1 * np.sin(t1)]).T

    r2 = np.sqrt(3 * generator.rand(args.points) + 0.9)
    t2 = 2 * np.pi * generator.rand(args.points)
    data2 = np.vstack([r2 * np.cos(t2), r2 * np.sin(t2)]).T

    labels = np.zeros([args.points * 2], dtype=int)
    labels[args.points :] = 1

    return np.vstack([data1, data2]), labels

def boundaryVisualization(args : argparse.Namespace, data : np.ndarray, labels : np.ndarray, generator : np.random.RandomState):
    colours = np.asarray(['red', 'blue'])
    if args.show_generated:
        _, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={'aspect': 'equal'})
        ax.scatter(data[:, 0], data[:, 1], marker='o', c=colours[labels])
        plt.tight_layout()
        plt.show()

    # TODO: Train the SVM classifier with your parameters.
    # - Try different parameters:
    #   - C > 0 - Soft margin penalty
    #   - kernel can be one of ["rbf", "poly", "linear"], poly for polynomial
    #   - degree of polynomial kernel > 0 [int]
    #   - gamma describes the reach of influence of individual samples. Lower gamma means
    #     far reach, higher value means close reach. It can be "scale", "auto" or [float].
    gamma = args.gamma if args.gamma in ["scale", "auto"] else float(args.gamma)
    clf = sklearn.svm.SVC(C=args.C, kernel=args.kernel, degree=args.degree, gamma=gamma, random_state=generator)
    clf.fit(data, labels)

    # Create meshgrid data and predict it using the trained classifier.
    xDots = np.linspace(np.min(data[:, 0]) - 0.5, np.max(data[:, 0]) + 0.5, 150)
    yDots = np.linspace(np.min(data[:, 1]) - 0.5, np.max(data[:, 1]) + 0.5, 150)
    xx, yy = np.meshgrid(xDots, yDots)
    meshData = np.vstack([xx.ravel(), yy.ravel()]).T

    predictions = clf.predict(meshData)

    # Draw the decision boundary of the classifier.
    _, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={'aspect': 'equal'})
    ax.pcolormesh(xx, yy, predictions.reshape(xx.shape), cmap=ListedColormap(colours), alpha=0.25, shading="auto")
    ax.scatter(data[:, 0], data[:, 1], marker='o', c=colours[labels])

    decGrid = clf.decision_function(meshData)
    ax.contour(xx, yy, decGrid.reshape(xx.shape), colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    decision_function = clf.decision_function(data)
    support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]

    ax.scatter(data[support_vector_indices, 0], data[support_vector_indices ,1], edgecolors='k', marker='o', facecolors='none', s=80, linewidths=0.5)
    plt.tight_layout()
    plt.show()

def crossValidation(args : argparse.Namespace, data : np.ndarray, labels : np.ndarray, generator : np.random.RandomState):
    gamma = args.gamma if args.gamma in ["scale", "auto"] else float(args.gamma)

    # TODO: Use crossvalidation to find the best parameters and to compare the performance of
    # different kernels. (If you don't have a lot of time, go look at 'lab08d' and solve multi-class
    # classification problem before playing with cross-validation.)
    # - You may use sklearn.model_selection.GridSearchCV but this exercise is focused on comparing
    #   the classification performance of SVM with differnet kernels/parameters rather than just finding
    #   the best one.
    # - Train several models and evaluate them on validation data.
    # - Print the performance (accuracy/error rate) of all examined models.
    
    kf = sklearn.model_selection.KFold(args.kfold_split, shuffle=True, random_state=generator)
    for trainIndices, validationIndices in kf.split(data):
        pass

def main(args : argparse.Namespace):
    # NOTE: Kernel SVM exercise.

    generator = np.random.RandomState(args.seed)

    # TODO: Data generation - choose which one you want to use in the arguments.
    # - You can modify the data generating parameters if you want.
    if args.data == "mvn":
        data, labels = mvnData(generator, args)
    elif args.data == "circle":
        data, labels = circleData(generator, args)
    else:
        raise ValueError("Unknown data type.")

    tasks = {
        "visualize" : boundaryVisualization,
        "crossval" : crossValidation,
    }
    if args.task not in tasks.keys():
        raise ValueError("Unknown task: '{}'.".format(args.task))
    tasks[args.task](args, data, labels, generator)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

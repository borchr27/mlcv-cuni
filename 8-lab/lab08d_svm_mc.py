
import argparse
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import sklearn.svm
import lab08_help


parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--seed", default=None, type=int, help="Seed for data generation.")
parser.add_argument("--points", default=100, type=int, help="Number of points generated for each class.")
parser.add_argument("--show_generated", default=False, action="store_true", help="Plots the generated data before computation.")
parser.add_argument("--C", default=1.0, type=float, help="Soft margin constraint for SVM.")
parser.add_argument("--kernel", default="rbf", type=str, help="SVM kernel.")
parser.add_argument("--degree", default=3, type=int, help="Degree of polynomial kernel.")
parser.add_argument("--gamma", default="scale", help="Regularisation of sample influence.")
parser.add_argument("--multiclass", default="ovr", type=str, help="Multi-class classification strategy.")


def main(args : argparse.Namespace):
    # NOTE: Multiclass SVM exercises.

    generator = np.random.RandomState(args.seed)

    # Data generation - You can add more classes or change the parameters if you want to do so.
    mu1 = [1, 1]
    sigma1 = [[0.5, 0], [0, 0.25]]
    mu2 = [6, 2]
    sigma2 = [[0.5, 0], [0, 1]]
    mu3 = [1, 4]
    sigma3 = [[0.5, 0], [0, 0.25]]
    mu4 = [4, 3.5]
    sigma4 = [[0.75, 0.1], [0.1, 0.4]]

    labelValues = [0, 1, 2, 3]
    numClasses = len(labelValues)

    data, labels = lab08_help.mvnGenerateData(
        generator, args.points,
        [mu1, mu2, mu3, mu4],
        [sigma1, sigma2, sigma3, sigma4],
        labelValues
    )

    colours = np.asarray(['red', 'blue', 'green', 'cyan', 'magenta', 'yellow'])
    if args.show_generated:
        _, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={'aspect': 'equal'})
        ax.scatter(data[:, 0], data[:, 1], marker='o', c=colours[labels])
        plt.tight_layout()
        plt.show()

    # TODO: Train SVM classifiers (SVC) for multi-class classification.
    # Use the arguments 'args.C', 'args.kernel', 'args.degree' and 'args.gamma' in the instantiation
    # of the classifier so that you can easily examine the effect of the parameters in the multi-class
    # problem.
    #
    # If 'args.multiclass == "ovr"' then:
    # - Train classifier for every class by temporarily changing the labels
    #   to 1 for the particular class and 0 for all other classes.
    # - Store the trained models.
    # If 'args.multiclass == "ovo"' then:
    # - Train a classifier for every pair of classes. You have to select only the data from the two
    #   selected classes and no other.
    # - Store the trained models and make sure that you know which model predicts for which classes.
    # If 'args.multiclass == "builtin"' then:
    # - Train one SVM classifier using 'data' and 'labels' without modification, the model will predict
    #   for multiple classes using one-vs-one ('ovo') strategy.
    #
    # NOTE: Use 'labelValues' list for manipulation of labels - it contains the class ID and every label
    # has one of the values from this list. For example, if you remember the index of a class 'clIdx' then
    # the class name/ID can be retrieved as 'labelValues[clIdx]'
    # NOTE: To compute logical operations on boolean arrays, you can use 'np.logical_or' (and/not...)



    # TODO: Create meshgrid as in previous exercises and predict the classes of meshgrid points
    # using the trained classifiers.
    #
    # - For 'args.multiclass == "ovr"' select the class as in 'lab08a' by finding the first
    #   positive classification and defaulting to the first class if there are none.
    #   NOTE: Other models (and with some modifications SVM as well) produce probabilities
    #   in classification and you can choose the most probable one, which often avoids 'weird'
    #   decision boundaries in the ovr case.
    # - For 'args.multiclass == "ovo"' use majority voting with 'np.argmax' to select the most
    #   commonly predicted class for every point.
    #   NOTE: You have to count the number of times a certain class was predicted by the 1vs1 models.
    # - For 'args.multiclass == "builtin"' use the predictions given by 'clf.predict' directly.
    #   NOTE: The results should be very similar to the 'ovo' implementation.

    xDots = np.linspace(np.min(data[:, 0]) - 0.5, np.max(data[:, 0]) + 0.5, 150)
    yDots = np.linspace(np.min(data[:, 1]) - 0.5, np.max(data[:, 1]) + 0.5, 150)
    xx, yy = np.meshgrid(xDots, yDots)
    meshData = np.vstack([xx.ravel(), yy.ravel()]).T

    predictions = None

    # TODO: Generate test data using 'lab08help.mvnGenerateData' the same way as it was used to generate
    # training data at the begining. Compute the classification accuracy.
    # Examine the performance (accuracies) of SVMs with different parameters and with 'ovo' and 'ovr' multiclass
    # classification strategies.
    
    # Visualise the decision boundaries.
    _, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={'aspect': 'equal'})
    ax.set_title("Decision boundaries with training data")
    ax.pcolormesh(xx, yy, predictions.reshape(xx.shape), cmap=ListedColormap(colours[:numClasses]), alpha=0.25, shading="auto")
    ax.scatter(data[:, 0], data[:, 1], marker='o', c=colours[labels])
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

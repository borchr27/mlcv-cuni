
import argparse
import numpy as np
import sklearn.svm
import sklearn.metrics
import lab08_help


parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--seed", default=None, type=int, help="Seed for RNG.")
parser.add_argument("--samples", default=3000, type=int, help="Number of images selected for training.")
parser.add_argument("--C", default=1.0, type=float, help="Soft margin constraint for SVM.")
parser.add_argument("--kernel", default="rbf", type=str, help="SVM kernel.")
parser.add_argument("--degree", default=3, type=int, help="Degree of polynomial kernel.")
parser.add_argument("--gamma", default="scale", help="Regularisation of sample influence.")
parser.add_argument("--multiclass", default="ovo", type=str, help="Multi-class classification strategy.")


def main(args : argparse.Namespace):
    # NOTE: SVM classification of MNIST exercise.

    # TODO: Load your MNIST data.
    train = lab08_help.MnistDataset("mnist_train.npz")
    test = lab08_help.MnistDataset("mnist_test.npz")

    # TODO: Train SVM using parameters specified in the arguments on the MNIST training set.
    # - Select only a subset of the training dataset (e.g. 2-5k samples so that it trains reasonably fast.)
    #   - The images are not ordered according to their labels so you do not have to permute them.



    # TODO: Evaluate the performance of the model by computing the error rate, accuracy
    # and F1 score on the MNIST test set.
    # NOTE: 'sklearn.metrics' has methods 'accuracy_score', 'f1_score'
    # which compute exactly what we want from the true labels and predictions.
    # - use average="macro" parameter for the 'f1_score' function to compute
    # the macro averaged score for multiclass classification.



    # TODO: Try training and evaluating the model only for classes '3' and '5', '4' and '9' or '1' and '7',
    # which are the most confused pairs of digits in the MNIST dataset.

    pass

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

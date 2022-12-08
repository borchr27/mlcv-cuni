
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sklearn.tree
from sklearn.metrics import accuracy_score
import lab09_help


parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--seed", default=None, type=int, help="Seed for RNG.")
parser.add_argument("--threshold", default=0.5, type=float, help="Threshold for MNIST binarisation.")
parser.add_argument("--samples", default=5000, type=int, help="Number of used training samples.")
parser.add_argument("--bootstrap_samples", default=30, type=int, help="Number of bootstrap samples.")
parser.add_argument("--max_depth", default=None, type=int, help="The maximum depth of the trained trees.")
parser.add_argument("--min_samples_split", default=10, type=int, help="The minimum number of samples required fro a node split.")
parser.add_argument("--min_samples_leaf", default=1, type=int, help="The minimum number of samples, which has to be present in a leaf.")
parser.add_argument("--max_leaf_nodes", default=None, type=int, help="The maximum number of leaf nodes allowed in the tree.")


def main(args : argparse.Namespace):
    generator = np.random.RandomState(args.seed)

    # TODO: Load the MNIST data and binarise them using 'args.threshold'.
    train = lab09_help.MnistDataset("mnist_train.npz")

    train.imgs = np.asarray(train.imgs > args.threshold, np.int32)

    # TODO: Select only a subset of the training data defined by 'args.samples'.
    trainData = None
    trainLabels = None

    # Let's apply bagging on decision trees.

    # Create an array for class association counting.
    classifCount = np.zeros([args.samples, 10], dtype=np.int32)

    # Create 'args.bootstrap_samples' bootstrap samples.
    dataRange = np.arange(args.samples)
    err = np.zeros([args.bootstrap_samples])

    for sample in range(args.bootstrap_samples):
        # Sample data with replacement.
        bootstrapSample = generator.choice(args.samples, args.samples, replace=True)
        # TODO: Train a DecisionTreeClassifier with parameters from args and data
        # given by 'bootstrap_sample'.
        tree = None

        # Find OOB data samples.
        oobDataIndices = np.setdiff1d(dataRange, bootstrapSample)
        # TODO: Use 'tree' to predict data given by 'oobDataIndices' and add 1 to elements
        # of 'classifCount' matrix specified by 'oobDataIndices' and the computed predictions.
        predictions = None

        # TODO: Compute the OOB classification from 'classifCount'.
        classifOob = None

        # TODO: Compute ensemble OOB error rate (from predictions given by 'classifOob').
        err[sample] = None
        print("OOB error rate of an ensemble with {:>3} classifiers is {:.2f}%".format(sample + 1, err[sample] * 100))

    # Visualise the OOB error rate.
    _, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={'aspect': 'auto'})
    ax.step(np.arange(args.bootstrap_samples), err, where="post")
    ax.set_title("Ensemble OOB error rate.")
    ax.set_xlabel("Size of the ensemble.")
    ax.set_ylabel("OOB error rate.")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

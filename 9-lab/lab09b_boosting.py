
import argparse
import numpy as np
import sklearn.tree
from sklearn.metrics import accuracy_score
import lab09_help


parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--seed", default=None, type=int, help="Seed for RNG.")
parser.add_argument("--stumps", default=50, type=int, help="Number of bootstrap samples.")
parser.add_argument("--max_depth", default=None, type=int, help="The maximum depth of the trained trees.")
parser.add_argument("--min_samples_split", default=10, type=int, help="The minimum number of samples required fro a node split.")
parser.add_argument("--min_samples_leaf", default=1, type=int, help="The minimum number of samples, which has to be present in a leaf.")
parser.add_argument("--max_leaf_nodes", default=None, type=int, help="The maximum number of leaf nodes allowed in the tree.")


def main(args : argparse.Namespace):
    generator = np.random.RandomState(args.seed)

    # TODO: Load the MNIST data.
    train = None
    test = None

    # TODO: Create training dataset consisting only of 3s and 5s (the most confused pair of digits).
    trainData = None
    # TODO: Make sure that the classes are +1 and -1.
    # NOTE: 'train.labels' is uint8 array, so you will have to convert to a signed data type if you want
    # to use it in some mathematical operation.
    trainLabels = None
    # Assign the number of training samples to 'train_size'
    trainSize = trainData.shape[0]

    # TODO: Implement Adaboost algorithm
    # - use decision stumps (trees with 2 leafs) as weak classifiers.

    # TODO: Initialise the weight array.
    weights = None
    # TODO: Initialise the alpha value to 0 for every stump.
    alphas = None

    stumps = []
    for stumpIdx in range(args.stumps):
        stumps.append(sklearn.tree.DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_split=trainSize, # Ensure small trees.
            min_samples_leaf=args.min_samples_leaf,
            max_leaf_nodes=args.max_leaf_nodes,
            random_state=generator
        ))
        stumps[stumpIdx].fit(trainData, trainLabels, sample_weight=weights)

        # TODO: Compute predictions of the training data for the current stump.
        trainPredictions = None

        # TODO: Compute the training error of the current stump weighted by the up-to-date weights.
        # - NOTE: We are using custom weights. Until now we considered each sample to have the same weight.
        err = None
        # TODO: Compute the alpha value for this stump.
        alphas[stumpIdx] = None
        print("Error rate: {:7.4f}   with alpha {:7.4f}".format(err, alphas[stumpIdx]))

        # TODO: Update the weights.
        weights = None

    # Evaluation on the test set.
    # TODO: Select and modify the test data the same way as the training data at the beginning.
    testData = None
    # TODO: Make sure that the classes are -1 and 1.
    testLabels = None
    testSize = testData.shape[0]

    # TODO: Compute the predictions of the boosted model. To get sign of values, use 'np.sign'.
    predictions = None

    # Let's look at the accuracy of the boosted model.
    testAccuracy = accuracy_score(testLabels, predictions)
    print("Test error rate is {:.2f}%.".format((1.0 - testAccuracy) * 100))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

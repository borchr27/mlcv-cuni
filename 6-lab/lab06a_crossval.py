
import argparse
import numpy as np
import sklearn.model_selection
import sklearn.naive_bayes
import lab06_help
from scipy.stats import norm

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="er", type=str, help="Executed task, one of: 'er', 'tuning'.")
parser.add_argument("--threshold", default=0.5, type=float, help="Threshold for image binarisation.")
parser.add_argument("--alpha", default=1, type=float, help="Bernoulli naive bayes laplace smoothing factor - can be used as a hyper-parameter.")
parser.add_argument("--confidence_alpha", default=0.05, type=float, help="The alpha value considered when working with confidence intervals.")
parser.add_argument("--kfold_split", default=5, type=int, help="Number of cross validation splits.")
parser.add_argument("--num_searched_thresholds", default=10, type=int, help="Number of thresholds searched during parameter tuning.")

class BayesClassifier:
    # TODO: Implement the bernoulli naive bayes classifier for MNIST dataset.
    # Ideally, you should take your solution to lab04a.py and use it in this class.
    # - If you don't have the solution or you are not confident in it's performance, you may
    #   use sklearn.naive_bayes.BernoulliNB and its fit() and predict() methods, which are
    #   equivalent to the methods fit() and predict() of this class.

    # TODO: Use the init function to initialise your classifier hyperparameters, such as
    # Laplace smoothing factor alpha.
    def __init__(self, args : argparse.Namespace) -> None:
        self._alpha = args.alpha

    # TODO: Implement the training routine.
    def fit(self, imgTrain : np.ndarray, labelTrain : np.ndarray) -> None:
        raise NotImplementedError()

    # TODO: Implement the predicting routine.
    # - NOTE: This function should return final classifications. The 'predict' method in lab04a_nb_bin.py returned probabilities.
    def predict(self, imgTest : np.ndarray) -> np.ndarray:
        raise NotImplementedError()

def errorRate(labelTrue : np.ndarray, labelPredict : np.ndarray) -> float:
    # TODO: Return the error rate computed from the given predicted and true labels, i.e. 1 - accuracy.
    raise NotImplementedError()

def compareErrorRate(train : lab06_help.MnistDataset, test : lab06_help.MnistDataset, args : argparse.Namespace):
    # TODO: Implement the BernoulliNB class, such that it is able to train and predict
    # MNIST data using bernoulli (binary) naive bayes from Lesson 3.
    #
    # -> Choose a threshold for conversion to binary images (args.threshold).
    # -> Train the Bayes classifier on all training data (no test data).


    # TODO: Compute confidence intervals:
    # -> Implement function errorRate() which computes the error rate of the given classification
    #    results.
    # -> Compute the error rate on MNIST test data (test images).
    # -> Compute the confidence intervals of the true error.
    # -> For critical value of the normal distribution, you can use the table from the lecture slides or function 'norm.ppf'.


    # TODO: Complete the cross-validation code in the following for loop.
    # -> Do not forget to binarise images.
    # -> We are performing 5-fold cross validation on the training data to get the estimate of the error rate on unseen data.

    kf = sklearn.model_selection.KFold(args.kfold_split)
    err = np.zeros([kf.get_n_splits()])

    for trainIndices, validationIndices in kf.split(train.imgs):
        # Indices for our training and validation sets.
        # - 'trainIndices', 'validationIndices'

        # TODO: Compute the error for this split on validation data.
        # -> Train NB classifier on data selected by 'trainIndices'.
        # -> Compute prediction on the data given by 'validationIndices'.
        # -> Compute the error rate using the function 'errorRate()' and store it in 'err'.
        pass

    
    # TODO: Compute the mean error of validation splits.


    # TODO: Compare with the error computed on the test dataset when we trained the classifier on
    # the whole training set.

    pass

def parameterTuning(train : lab06_help.MnistDataset, test : lab06_help.MnistDataset, args : argparse.Namespace):
    # TODO: Parameter tuning
    # -> Perform 5-fold cross validation on MNIST training data to obtain the best binarisation threshold value.
    #    Use total error rate, i.e. the mean of validation error rates, as the score.

    pass

def main(args : argparse.Namespace):
    # TODO: Load MNIST data using lab06_help.MnistDataset class (or any other way you see fit == will need a function signature update).
    # -> If you use the data provided in moodle then both datasets will have attributes 'imgs' for images and 'labels' for labels.
    train = lab06_help.MnistDataset("mnist_train.npz")
    test = lab06_help.MnistDataset("mnist_test.npz")

    # Execute task 'args.task'.
    tasks = {
        "er" : compareErrorRate,
        "tuning" : parameterTuning
    }
    if args.task not in tasks.keys():
        raise ValueError("Unrecognised task: '{}'".format(args.task))
    tasks[args.task](train, test, args)
    

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

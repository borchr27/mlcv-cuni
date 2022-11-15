
import argparse
import numpy as np
import lab06_help

# NOTE: This exercise aims to compare differences in error between NB and KNN classifiers using hypothesis testing.
# It is a voluntary part of the exercise (for bonus points, that might help you at the exam).
# - Hand in the code that test hypotheses using the four described tests.
# - Summarise the results in written text and submit it with the code.

parser = argparse.ArgumentParser()
parser.add_argument("--x_scatter", default=0.25, type=float, help="X scatter of the generated data.")
parser.add_argument("--seed", default=None, type=int, help="Seed for RNG in this assignment.")
# You may add any arguments you want, the following ones are my recommendation.
parser.add_argument("--test", default="basic", type=str, help="Which statistic do we want to evaluate: 'basic', 'paired', 'corrected', 'mcnemar'.")
parser.add_argument("--train_size", default=200, type=int, help="Number of points in each class in the train set.")
parser.add_argument("--test_size", default=150, type=int, help="Number of points in each class in the test set.")
parser.add_argument("--k_neighbors", default=9, type=int, help="Number of points for KNN classification.")
parser.add_argument("--paired_k_splits", default=3, type=int, help="Number of splits used in the paired t-test.")
parser.add_argument("--corrected_kfold_splits", default=3, type=int, help="Number of folds considered in the corrected resampled t-test.")
parser.add_argument("--confidence_alpha", default=0.05, type=float, help="The alpha value considered when working with confidence of hypothesis rejection.")

def basicTest(args : argparse.Namespace, generator : np.random.RandomState):
    # TODO: Basic test trained on independent samples and tested on independent samples.
    raise NotImplementedError()

def pairedTTest(args : argparse.Namespace, generator : np.random.RandomState):
    # TODO: Paired t-test.
    raise NotImplementedError()

def correctedResampledTTest(args : argparse.Namespace, generator : np.random.RandomState):
    # TODO: Corrected resampled t-test.
    raise NotImplementedError()

def mcNemarTest(args : argparse.Namespace, generator : np.random.RandomState):
    # TODO: McNemar's test.
    raise NotImplementedError()

def main(args : argparse.Namespace):
    # NOTE: Since you are eligible to gain bonus points from this exercise, you should use
    # algorithms and code structures you learnt in the previous practicals.
    # - i.e. this assignment is pretty bare-bone.

    # Use the following generator for all RNG in your code.
    generator = np.random.RandomState(args.seed)

    # NOTE: Scikit-learn classifiers
    # - NB ... sklearn.naive_bayes.GaussianNB
    # - KNN ... sklearn.neighbors.KNeighborsClassifier

    # TODO: Set the parameters of the classifiers.
    # TODO: Evaluate the hypothesis that both classifiers have the same error.
    #
    # 1) Train NB and KNN on independent samples, test on independent samples.
    # 2) Do the paired t-test (train on independent samples, test on paired data).
    # 3) Do the corrected resampled t-test.
    # 4) Do the McNemar's test.
    #
    # - To generate training/testing data, use 'lab06_help.generateData' with arguments from args.
    #   - e.g. lab06_help.generateData(generator, args.train_size, args.x_scatter)
    # - To compute the critical value of Normal distribution, use scipy.stats.norm.ppf
    # - To compute the critical value of Student's distribution, use scipy.stats.t.ppf
    # - To compute the critical value of Chi2 (Chi squared) distribution, use scipy.stats.chi2.ppf

    tests = {
        "basic" : basicTest,
        "paired" : pairedTTest,
        "corrected" : correctedResampledTTest,
        "mcnemar" : mcNemarTest
    }
    if args.test not in tests.keys():
        raise ValueError("Unrecognised hypothesis testing mode: '{}'!".format(args.test))
    tests[args.test](args, generator)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

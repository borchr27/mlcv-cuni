
import argparse
from typing import Tuple, Sequence, List
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import sklearn.metrics
import lab05_help
import lab05b_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=None, type=int, help="Seed for the random classifier.")
parser.add_argument("--num_thresholds", default=11, type=int, help="Number of thresholds for linearly-spaced curve plotting.")
# TODO: Use 'test' argument to run the individual tests. First, complete 'pr', 'roc', 'f1' and then 'ap'.
parser.add_argument("--test", default="pr", type=str, help="Investigated curve, one of: 'pr', 'roc', 'f1', 'ap'.")

def computeThresholds(thresholdType : str, args: argparse.Namespace, classifierData : np.ndarray = None) -> List[np.ndarray]:
    if thresholdType == "linspace":
        # TODO: Compute linearly spaced thresholds for each classifier.
        # - Use 'args.num_thresholds' as the number of linearly-spaced thresholds.
        # - Return a 'list' of 'np.ndarrray' thresholds.
        raise NotImplementedError()
    elif thresholdType == "exact":
        # TODO: Compute exact threhsold values for each classifier.
        # - Exact thresholds means that you should return all values where the classifier probabilities change in the ascending order.
        # - Return a 'list' of 'np.ndarray' thresholds.
        raise NotImplementedError()
    else:
        raise ValueError("Unrecognised thresholdType: '{}'".format(thresholdType))

def computeMetrics(thresholds : Sequence[np.ndarray], classifierData : np.ndarray) -> Tuple[List[np.ndarray], ...]:
    # TODO: Use 'lab05_help.getConfusionMatrix' and 'lab05b_metrics.classifierMetrics' to compute false positive rate, true positive
    # rate, positive predictive value and f1 score for each classifier at the given thresholds.
    # The result should be a List of numpy arrays for each metric, e.g.,
    # - 'fprs' := [randomClassifierFprs, firstClassifierFprs, secondClassifierFprs, thirdClassifierFprs, fourthClassifierFprs, fifthClassifierFprs]
    # - same for 'tprs', 'ppvs', 'f1s'.
    fprs = None
    tprs = None
    ppvs = None
    f1s = None
    return fprs, tprs, ppvs, f1s

def computeAAP(precisions : np.ndarray, recalls : np.ndarray) -> Tuple[np.ndarray, float]:
    # TODO: Compute approximated average precision.
    # Return new precision values 'apPrecisions' (for graph plotting) and the computed AAP value 'aap'.
    apPrecisions = None
    aap = None
    return apPrecisions, aap

def computeIAP(precisions : np.ndarray, recalls : np.ndarray) -> float:
    # TODO: Compute interpolated average precision.
    # Return new precision values 'apPrecisions' (for graph plotting) and the computed IAP value 'iap'.
    apPrecisions = None
    iap = None
    return apPrecisions, iap

def testAP(precisionsLinspace : Sequence[np.ndarray], recallsLinspace : Sequence[np.ndarray], precisionsExact : Sequence[np.ndarray], recallsExact : Sequence[np.ndarray]):
    # TODO: Finish functions 'computeAAP' and 'computeIAP'.
    # This test will evaluate AAP and IAP for the classifier at index 'clfIdx' from linearly-spaced and exact thresholds.
    clfIdx = 1
    aapPrecisionsLin, aapLin = computeAAP(precisionsLinspace[clfIdx], recallsLinspace[clfIdx])
    aapPrecisionsEx, aapEx = computeAAP(precisionsExact[clfIdx], recallsExact[clfIdx])
    iapPrecisionsLin, iapLin = computeIAP(precisionsLinspace[clfIdx], recallsLinspace[clfIdx])
    iapPrecisionsEx, iapEx = computeIAP(precisionsExact[clfIdx], recallsExact[clfIdx])

    _, ax = plt.subplots(1, 4, figsize=(16, 4), subplot_kw={'aspect': 'equal'})
    lab05_help.drawAPPlot(ax[0], precisionsLinspace[clfIdx], aapPrecisionsLin, recallsLinspace[clfIdx], aapLin, "AAP Linspace")
    lab05_help.drawAPPlot(ax[1], precisionsExact[clfIdx], aapPrecisionsEx, recallsExact[clfIdx], aapEx, "AAP Exact")
    lab05_help.drawAPPlot(ax[2], precisionsLinspace[clfIdx], iapPrecisionsLin, recallsLinspace[clfIdx], iapLin, "IAP Linspace")
    lab05_help.drawAPPlot(ax[3], precisionsExact[clfIdx], iapPrecisionsEx, recallsExact[clfIdx], iapEx, "IAP Exact")
    plt.tight_layout()
    plt.show()

def testPR(precisionsLinspace : Sequence[np.ndarray], recallsLinspace : Sequence[np.ndarray], precisionsExact : Sequence[np.ndarray], recallsExact : Sequence[np.ndarray], classifierData : np.ndarray):
    # TODO: Complete this function by filling in plotting code for linearly-spaced, exact and sklearn PR curve.
    # NOTE:
    # - 'exact' and 'sklearn' curves should be exactly the same.
    # - Why is the curve based on linearly-spaced thresholds different?
    _, ax = plt.subplots(1, 3, figsize=(14, 5), subplot_kw={'aspect': 'equal'})
    names = ["Random", "Clf 1", "Clf 2", "Clf 3", "Clf 4", "Clf 5"] # Use these names as plot legend labels.
    # TODO: Plot (Axes.plot) linspace PR into ax[0] for all 6 classifiers.
    
    # TODO: Plot (Axes.plot) exact PR into ax[1] for all 6 classifiers.
    
    # TODO: Plot (Axes.plot) sklearn PR into ax[2] for all 6 classifiers.
    # - Use 'sklearn.metrics.precision_recall_curve' to compute precisions and recalls in scikit-learn.
    
    lab05_help.setAxes(ax[0], "PR curve (linspace)", "Recall", "Precision")
    lab05_help.setAxes(ax[1], "PR curve (exact)", "Recall", "Precision")
    lab05_help.setAxes(ax[2], "PR curve (sklearn)", "Recall", "Precision")
    plt.tight_layout()
    plt.show()

def testROC(fprsLinspace : Sequence[np.ndarray], tprsLinspace : Sequence[np.ndarray], fprsExact : Sequence[np.ndarray], tprsExact : Sequence[np.ndarray], classifierData : np.ndarray):
    # TODO: Complete this function by filling in plotting code for linearly-spaced, exact and sklearn ROC curve.
    # NOTE:
    # - 'exact' and 'sklearn' curves should be exactly the same.
    # - Why is the curve based on linearly-spaced thresholds different?
    _, ax = plt.subplots(1, 3, figsize=(14, 5), subplot_kw={'aspect': 'equal'})
    names = ["Random", "Clf 1", "Clf 2", "Clf 3", "Clf 4", "Clf 5"]
    # TODO: Plot (Axes.plot) linspace ROC into ax[0] for all 6 classifiers.
    
    # TODO: Plot (Axes.plot) exact ROC into ax[1] for all 6 classifiers.
    
    # TODO: Plot (Axes.plot) sklearn ROC into ax[2] for all 6 classifiers.
    # - Use 'sklearn.metrics.roc_curve' to compute fprs and tprs in scikit-learn.
    
    lab05_help.setAxes(ax[0], "ROC curve (linspace)", "FPR", "TPR")
    lab05_help.setAxes(ax[1], "ROC curve (exact)", "FPR", "TPR")
    lab05_help.setAxes(ax[2], "ROC curve (sklearn)", "FPR", "TPR")
    plt.tight_layout()
    plt.show()

def testF1(thresholdsLinspace : Sequence[np.ndarray], f1sLinspace : Sequence[np.ndarray], thresholdsExact : Sequence[np.ndarray], f1sExact : Sequence[np.ndarray]):
    # TODO: Complete this function by filling in plotting code for linearly-spaced and exact F1 curve.
    # - Why do the curves differ?
    # - Which threshold maximises the F1 score?
    _, ax = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'aspect': 'equal'})
    names = ["Random", "Clf 1", "Clf 2", "Clf 3", "Clf 4", "Clf 5"]
    # TODO: Plot (Axes.plot) linspace F1 curve into ax[0] for all 6 classifiers.
    
    # TODO: Plot (Axes.plot) exact F1 curve into ax[1] for all 6 classifiers.
    
    lab05_help.setAxes(ax[0], "F1 curve (linspace)", "Threshold", "F1")
    lab05_help.setAxes(ax[1], "F1 curve (exact)", "Threshold", "F1")
    plt.tight_layout()
    plt.show()

def main(args : argparse.Namespace):
    # Example data
    # Columns 0 - 4 are the output of five different classifiers (probability of belonging to the class 1).
    # Column 5 is the true class.
    rocData = scipy.io.loadmat("RocInput5.mat")
    rocData = rocData["RocInput5"]

    # Create a random classifier. You can change seed to make the results reproducible.
    generator = np.random.RandomState(args.seed)
    randomClassifier = generator.random([rocData.shape[0]])
    # Add the random classifier to the other ones so we can work with them in unified manner.
    # - NOTE: Now, column 0 is the random classifer, columns 1 - 5 are other classifiers and column 6 is the true class.
    classifierData = np.hstack((np.c_[randomClassifier], rocData))

    # Compute linearly-spaced and exact thresholds.
    thresholdsLinspace = computeThresholds("linspace", args, classifierData)
    thresholdsExact = computeThresholds("exact", args, classifierData)

    # Compute metrics at both threshold sets.
    fprsLin, tprsLin, ppvsLin, f1sLin = computeMetrics(thresholdsLinspace, classifierData)
    fprsEx, tprsEx, ppvsEx, f1sEx = computeMetrics(thresholdsExact, classifierData)

    if args.test == "pr":
        testPR(ppvsLin, tprsLin, ppvsEx, tprsEx, classifierData)
    elif args.test == "roc":
        testROC(fprsLin, tprsLin, fprsEx, tprsEx, classifierData)
    elif args.test == "f1":
        testF1(thresholdsLinspace, f1sLin, thresholdsExact, f1sEx)
    elif args.test == "ap":
        testAP(ppvsLin, tprsLin, ppvsEx, tprsEx)
    else:
        raise ValueError("Unrecognised test: '{}'".format(args.test))

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

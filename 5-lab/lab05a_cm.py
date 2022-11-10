
import scipy.io
import argparse
import lab05_help

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", default=0.75, type=float, help="Custom classification threshold.")
parser.add_argument("--classifier", default=0, type=int, help="Classifier index.")
parser.add_argument("--no_draw_matrix", default=False, action="store_true", help="Stops drawing confusion matrix.")
parser.add_argument("--log_color", default=False, action="store_true", help="Use logarithm when drawing colours of the confusion matrix.")

def main(args : argparse.Namespace):
    # Example of data with binary classification.
    # Columns 0 - 4 are the output of five different classifiers (probability of belonging to the class 1).
    # Column 5 is the true class.
    rocData = scipy.io.loadmat("RocInput5.mat")
    rocData = rocData["RocInput5"]

    # TODO: Choose the classification threshold 't' from [0, 1] (modify the argument or set 't' to a number).
    t = args.threshold

    # TODO: Choose which classifier to evaluate.
    classifIdx = args.classifier
    confusionMatrix = lab05_help.getConfusionMatrix(rocData[:, classifIdx], rocData[:, 5], t)

    # TODO: Inspect the matrix. Are the data balanced?

    # Draw the confusion matrix.
    if not args.no_draw_matrix:
        lab05_help.drawMatrices(confusionMatrix, args.log_color)

    # TODO: Choose different thresholds and compare confusion matrices.
    # - Compute mutliple confusion matrices using different thresholds and visualise them using 'lab05_help.drawMatrices'.
    # - You can pass a list of matrices to 'lab05_help.drawMatrices'.


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

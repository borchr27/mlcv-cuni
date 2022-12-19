
import argparse
import numpy as np
import scipy.io

parser = argparse.ArgumentParser()
parser.add_argument("--example", default="example", type=str, help="Example argument.")

class DataLoader:

    def __init__(self, filePath : str) -> None:
        # Load the matlab file.
        self.data = scipy.io.loadmat("segmentation_data.mat")
        # Extract the feature values.
        self.segmentationFeatures = self.data["segmentation_features"]
        # Extract the feature names - this is a nested array.
        self.featureNames = self.data["feature_names"]
        # Numerical labels in 1D array starting at index 0. 
        self.segmentationLabels = (self.data["segmentation_labels_num"] - 1).ravel()
        # Names of the classes from 0 to 6.
        self.classNames = ["BRICKFACE", "CEMENT", "FOLIAGE", "GRASS", "PATH", "SKY", "WINDOW"]

def main(args : argparse.Namespace):
    # NOTE: Description of the feature processing/classification/evaluation project part.
    #
    # The dataset given to you is in Matlab format and the PDF assignment is slightly
    # misleading because scipy cannot load categorical labels. Therefore, this is
    # a clarification of the input data description and the data contents.
    #
    # - Loading of .mat file through 'scipy.io.loadmat' results in a dictionary. It contains items:
    # - "segmentation_features" are a 2D numpy array as you would expect.
    # - "feature_names" is a nested array with string objects - you have to dig around to get a simple list of the names.
    # - "segmentation_labels" is broken - categorical labels are an enemy of scipy.
    # - "segmentation_labels_num" contain numerical labels starting at index 1.
    #
    # For your convenience, we defined class 'DataLoader' which extracts inforamtion from the .mat dictionary
    # in a more friendly format.
    data = DataLoader("segmentation_data.mat")

    raise NotImplementedError()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

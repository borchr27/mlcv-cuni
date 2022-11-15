
import os
import numpy as np
from typing import Sequence, Tuple, Union

def generateData(generator : np.random.RandomState, pointCount : Union[Sequence, int], xScatter : Union[Sequence, float], positions : Sequence = [1, 2, 3], scale : Sequence = [5, 5, 5]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates random data with custom point count and normal distribution in the X axis.
    Default parameters create three sets, each having a different label.
    Labels are marked from 0 to N - 1, where N is the number of generated sets.

    Arguments:
    - generator - Random number generator used for data generation.
    - pointCount - Number of points in each set. It can be a list or an int.
    - xScatter - Standard deviation in the X axis for the generated data. It can be a list or a float.
    - positions - Mean values of the generated sets. It has to be a sequence.
    - scale - Scale of the generated sets in the Y axis. It has to be a sequence.

    Returns:
    - xData - X coordinates of the generated data points (np.ndarray).
    - yData - Y coordinates of the generated data points (np.ndarray).
    - labels - Labels of the generated data points (np.ndarray).
    """
    size = len(positions) if len(positions) == len(scale) else None
    if size is None:
        raise ValueError("Incorrect size of input arrays.")
    
    pointCount = [pointCount] * size if isinstance(pointCount, int) else pointCount
    xScatter = [xScatter] * size if isinstance(xScatter, float) else xScatter
    xData = [positions[i] + xScatter[i] * generator.randn(pointCount[i]) for i in range(size)]
    xData = np.hstack(xData)

    yData = [scale[i] * generator.rand(pointCount[i]) for i in range(size)]
    yData = np.hstack(yData)

    labels = [np.ones(pointCount[i]) * (i + 1) for i in range(size)]
    labels = np.hstack(labels)

    return xData, yData, labels

class MnistDataset:
    """
    Loads the MNIST data saved in .npy or .npz files.

    If the 'labels' argument is left as None then the class assumes that the file
    in 'data' is .npz and creates attributes, with the same name as specified
    during the file creation, containing the respective numpy arrays.

    If the 'labels' argument is set to a string path then the class assumes that
    the files were saved as .npy and it will create two attributes: 'imgs' which
    contains the contents of the 'data' file and 'labels' with the contents of
    the 'labels' file.

    If you chose to save the arrays differently then you might have to modify
    this class or write your own loader.
    """

    def __init__(self, data : str = "mnist_train.npz", labels : str = None):

        if not os.path.exists(data):
            raise ValueError("Requested mnist data file not found!")
        if (labels is not None) and (not os.path.exists(labels)):
            raise ValueError("Requested mnist label file not found!")

        if labels is None:
            dataset = np.load(data)
            for key, value in dataset.items():
                setattr(self, key, value)
        else:
            self.imgs = np.load(data)
            self.labels = np.load(labels)

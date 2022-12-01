
import os
from typing import Tuple, Sequence
import numpy as np

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


def kPerceptronData(generator : np.random.RandomState, points : int, positions : np.ndarray, spread : float, clIds : Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates well separated data for K-class perceptron classification.

    Arguments:
    - 'generator' - Random number generator for reproducibility.
    - 'points' - Number of points generated for each class.
    - 'positions' - Positions of the class data blocks in the feature space.
    - 'spread' - Size of a class data block.
    - 'clIds' - Class label values, for three classes, it can be, for example, [1, 2, 3].

    Returns:
    - Data array with shape (points, 2).
    - Label array with shape (points,)
    """

    data = []
    classes = []
    for pos, clId in zip(positions, clIds):
        data.append(np.asarray(pos) + spread * generator.rand(points, 2))
        classes.append(clId * np.ones([points], dtype=int))
    data = np.vstack(data)
    classes = np.hstack(classes)
    return data, classes
    
def mvnGenerateData(generator : np.random.RandomState, points : int, mus : Sequence[np.ndarray], sigmas : Sequence[np.ndarray], labelMultipliers : Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates data from multivariate normal distribution.
    Expects lists of means 'mus' and covariance matrices 'sigmas' of the same length which
    define the distributions of the individual classes.
    The argument 'labelMultipliers' specifies the class numerical labels.

    Arguments:
    - 'generator' - Random number generator for reproducibility.
    - 'points' - Number of points generated for each class.
    - 'mus' - Means of the normally distributed classes.
    - 'sigmas' - Covariance matrices of the normally distributed classes.
    - 'labelMultipliers' - Class label values, for three classes, it can be, for example, [1, 2, 3].

    Returns:
    - Data array with shape (points, K) where K is the dimensionality of mu.
    - Label array with shape (points,)
    """
    
    if len(mus) != len(sigmas):
        raise ValueError("Data generation recieved different number of means than covariance matrices!")
    data = []
    labels = []
    for mu, sigma, labelMul in zip(mus, sigmas, labelMultipliers):
        data.append(generator.multivariate_normal(mu, sigma, size=points))
        labels.append(np.ones([points], dtype=int) * labelMul)
    
    return np.vstack(data), np.hstack(labels)

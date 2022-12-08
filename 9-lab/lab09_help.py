
import os
from typing import Sequence, Tuple
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

def generatePerceptronData(generator : np.random.RandomState, pointCount : int, shift : Sequence[np.ndarray], labelValues : Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates K normally distributed clusters where the length of 'shift' and 'labelValues' determines K.
    Arguments 'shift' and 'labelValues' have to have the same length.
    Depending on shift, the data will be linearly separable.

    Arguments:
    - 'generator' - Random number generator for reproducibility.
    - 'pointCount' - Number of points in each data cluster.
    - 'shift' - Additive displacement of each cluster.
    - 'labelValues' - Labels of the individual clusters.

    Returns:
    - Data array with shape (C * pointCount, 2) where C is the number of clusters.
    - Label array with shape (C * pointCount,) where C is the number of clusters.
    """
    points = []
    labels = []
    for i in range(len(shift)):
        points.append(generator.randn(pointCount, 2) + np.asarray(shift[i]))
        labels.append(np.ones([pointCount], np.int32) * labelValues[i])

    return np.vstack(points), np.hstack(labels)

def noisyXorData(generator : np.random.RandomState, pointCount : int, offset : float, labelValues : Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates XOR data with noise - It creates 4 clusters of data where
    the diagonal clusters are of the same class (1st and 3rd "quadrant" have one class, 2nd and 4th "quadrant" have another class).

    Arguments:
    - 'generator' - Random number generator for reproducibility.
    - 'pointCount' - Number of points in each cluster (Total number of points is 4 * 'pointCount').
    - 'offset' - Distance of data clusters from the origin.
    - 'labelValues' - Numbers representing labels of the data.

    Returns:
    - Data array with shape (4 * pointCount, 2).
    - Label array with shape (4 * pointCount,)
    """
    points = [
        np.vstack([generator.rand(pointCount) / 2 - offset, generator.rand(pointCount) / 2 + offset]).T,
        np.vstack([generator.rand(pointCount) / 2 + offset, generator.rand(pointCount) / 2 + offset]).T,
        np.vstack([generator.rand(pointCount) / 2 + offset, generator.rand(pointCount) / 2 - offset]).T,
        np.vstack([generator.rand(pointCount) / 2 - offset, generator.rand(pointCount) / 2 - offset]).T
    ]
    a = labelValues[0]
    b = labelValues[1]
    labels = [[a] * pointCount, [b] * pointCount, [a] * pointCount, [b] * pointCount]
    return np.vstack(points), np.hstack(labels)

def mvnGenerateData(generator : np.random.RandomState, pointCount : int, mus : Sequence[np.ndarray], sigmas : Sequence[np.ndarray], labelMultipliers : Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
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
    - Data array with shape (C * pointCount, K) where K is the dimensionality of mu and C is the number of clusters..
    - Label array with shape (C * pointCount,) where C is the number of clusters.
    """
    
    if len(mus) != len(sigmas):
        raise ValueError("Data generation recieved different number of means than covariance matrices!")
    data = []
    labels = []
    for mu, sigma, labelMul in zip(mus, sigmas, labelMultipliers):
        data.append(generator.multivariate_normal(mu, sigma, size=pointCount))
        labels.append(np.ones([pointCount], dtype=int) * labelMul)
    
    return np.vstack(data), np.hstack(labels)

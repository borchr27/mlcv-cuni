
import os
from typing import Sequence, Tuple
import numpy as np
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt

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

def generateSquareClusters(pointCount : int, offset : float, generator : np.random.RandomState) -> np.ndarray:
    """
    Generates 4 clusters in the corners of a square.
    The clusters are well-separable (depending on 'offset') and intended for clustering exercises.

    Arguments:
    - 'pointCount' - Number of points generated in each cluster.
    - 'offset' - Distance of each cluster from the origin of the coordiante system.
    - 'generator' - Random number generator for reproducibility.

    Returns:
    - Numpy ndarray with the shape (4 * pointCount, 2) of generated points.
    """
    data = [
        np.hstack([generator.rand(pointCount) - offset, generator.rand(pointCount) + offset, generator.rand(pointCount) + offset, generator.rand(pointCount) - offset]),
        np.hstack([generator.rand(pointCount) + offset, generator.rand(pointCount) + offset, generator.rand(pointCount) - offset, generator.rand(pointCount) - offset])
    ]
    return np.asarray(data).T

def mvnGenerateData(generator : np.random.RandomState, pointCount : int, mus : Sequence[np.ndarray], sigmas : Sequence[np.ndarray], labelMultipliers : Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates data from multivariate normal distribution.
    Expects lists of means 'mus' and covariance matrices 'sigmas' of the same length which
    define the distributions of the individual classes.
    The argument 'labelMultipliers' specifies the class numerical labels.
    The numbers of mus, sigmas and labelMultipliers have to match.

    Arguments:
    - 'generator' - Random number generator for reproducibility.
    - 'pointCount' - Number of points generated in each cluster.
    - 'mus' - Mean vectors of the multivariate normal distribution.
    - 'sigmas' - Covariance matrices of the multivariate normal distribution.
    - 'labelMultipliers' - Numerical label values of the individual classes.

    Returns:
    - Numpy ndarray of shape (len(mus) * pointCount, mus[0].size) with generated data.
    - Numpy ndarray of shape (len(mus) * pointCount,) with generated labels.
    """
    
    if len(mus) != len(sigmas):
        raise ValueError("Data generation recieved different number of means than covariance matrices!")
    data = []
    labels = []
    for mu, sigma, labelMul in zip(mus, sigmas, labelMultipliers):
        data.append(generator.multivariate_normal(mu, sigma, size=pointCount))
        labels.append(np.ones([pointCount], dtype=int) * labelMul)
    
    return np.vstack(data), np.hstack(labels)

def showSilhouette(data : np.ndarray, labels : np.ndarray, clusters : int, ax : plt.Axes = None) -> None:
    """
    Draws a silhouette graph - a bar with silhouette value for every data sample.

    Arguments:
    - 'data' - Feature array with the visualised data.
    - 'labels' - Labels of the given data points from 'data'.
    - 'clusters' - Number of discovered clusters.
    """
    sample_silhouette_values = silhouette_samples(data, labels)
    order = np.lexsort([-sample_silhouette_values, labels])
    indices = [np.nonzero(labels[order] == k)[0] for k in range(clusters)]
    ytick = [(np.max(indices[idx]) + np.min(indices[idx])) / 2 if indices[idx].size != 0 else np.max(np.concatenate(indices[:idx] + [[0]])) for idx in range(len(indices))]
    yticklabels = ["{}".format(k) for k in range(clusters)]

    colours = ["C{}".format(c) for c in labels[order]]
    fullDraw = ax is None
    if fullDraw:
        _, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={"aspect": "auto"})
    ax.barh(np.arange(data.shape[0]), sample_silhouette_values[order], height=1.0, edgecolor="none", color=colours)
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_yticks(ytick)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Silhouette value")
    ax.set_ylabel("Cluster")
    ax.set_title("Silhouette graph")
    if fullDraw:
        plt.tight_layout()
        plt.show()

def plotScores(clusterList : Sequence[int], scores : np.ndarray, title : str = "", xlabel : str = "", ylabel : str = "", ax : plt.Axes = None) -> None:
    """
    Plots a simple line graph of cluster values with custom labels.
    
    Arguments:
    - 'clusterList' - Sequence of integers represententing the number of clusters (The X axis of the graph).
    - 'scores' - Values shown on the Y axis of the graph. Has to have the same length as 'clusterList'
    - 'title' - Title of the axis.
    - 'xlabel' - Label written on the X axis.
    - 'ylabel' - Label written on the Y axis.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={"aspect": "auto"})
    ax.plot(clusterList, scores)
    ax.scatter(clusterList, scores)
    ax.set_xticks(clusterList)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ax is None:
        plt.tight_layout()
        plt.show()


import argparse
from typing import Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
import lab11_help

parser = argparse.ArgumentParser()
# General parameters.
parser.add_argument("--seed", default=None, type=int, help="Seed for RNG.")
parser.add_argument("--data", default="square", type=str, help="Which data should be used: 'square', 'normal', 'mnist'.")
parser.add_argument("--test", default=False, action="store_true", help="Run a no-error test trying to catch exceptions depending on RNG.")
# K-Medoids parameters.
parser.add_argument("--distance", default="squareEuclidean", type=str, help="Distance metric used in K-Medoids algorithm.")
parser.add_argument("--clusters", default=5, type=int, help="Number of requested clusters.")
parser.add_argument("--restarts", default=10, type=int, help="Number of restarts of the K-Medoids algorithm.")
parser.add_argument("--max_iter", default=300, type=int, help="Maximum number of iterations allowed in K-Medoids optimisation.")
parser.add_argument("--mean_centers", default=False, action="store_true", help="Computes euclidean means as centers instead of medoids.")
# Data parameters.
parser.add_argument("--points", default=200, type=int, help="Number of points to generate.")
parser.add_argument("--offset", default=1.1, type=float, help="Default offset of the square data.")
parser.add_argument("--mnist_count", default=500, type=int, help="Number of images taken from the MNIST dataset.")


def euclidean(a : np.ndarray, b : np.ndarray) -> np.ndarray:
    # NOTE: Expects 'a' to be a single vector, e.g. the center and 'b' as a data array.
    return np.sqrt(np.sum((a - b) ** 2, axis = -1))

def squareEuclidean(a : np.ndarray, b : np.ndarray) -> np.ndarray:
    # NOTE: Expects 'a' to be a single vector, e.g. the center and 'b' as a data array.
    return np.sum((a - b) ** 2, axis = -1)

def cosine(a : np.ndarray, b : np.ndarray) -> np.ndarray:
    # NOTE: Expects 'a' to be a single vector, e.g. the center and 'b' as a data array.
    a_norm = np.linalg.norm(a, axis = -1)
    b_norm = np.linalg.norm(b, axis = -1)
    similiarity = np.dot(a, b.T)/(a_norm * b_norm)
    dist = 1. - similiarity
    return dist

def manhattan(a : np.ndarray, b : np.ndarray) -> np.ndarray:
    # NOTE: Expects 'a' to be a single vector, e.g. the center and 'b' as a data array.
    return np.sum(np.abs(a - b), axis = -1)

def chebyshev(a : np.ndarray, b : np.ndarray) -> np.ndarray:
    # NOTE: Expects 'a' to be a single vector, e.g. the center and 'b' as a data array.
    return np.max(np.abs(a - b), axis = - 1)

def hamming(a : np.ndarray, b : np.ndarray) -> np.ndarray:
    """Hamming distance is the number of pixels in which the images differ."""
    # NOTE: Expects 'a' to be a single vector, e.g. the center and 'b' as a data array.
    # This distance metric doesn't make much sense for spatial data.
    a = a >= 0.5
    b = b >= 0.5
    return np.sum(a != b, axis = -1)


class NaiveKMedoids():
    """This class contains an implementation of naive K-Medoids algorithm."""

    def __init__(
            self, num_clusters : int, distance : Callable[[np.ndarray, np.ndarray], np.ndarray], 
            restarts : int = 10, max_iter : int = 300, tol : float = 0.001, 
            generator : np.random.RandomState = None, medoid : bool = True, verbose : bool = True) -> None:
        # NOTE: Initialisation is mainly about storing the given paramters.
        # - We denote private attributes by a leading underscore, e.g., 'self._restarts'
        self.num_clusters = num_clusters
        self._distance = distance
        self._restarts = restarts
        self._max_iter = max_iter
        self._tol = tol
        if generator is None or isinstance(generator, int):
            self._generator = np.random.RandomState(generator)
        else:
            self._generator = generator
        self._medoid = medoid
        self._verbose = verbose

    def _initialize(self, data : np.ndarray) -> np.ndarray:
        # TODO: Compute initial cluster centers from the data by selecting random 'self.num_clusters' data points.
        # - To compute indices without repetition, use 'self._generator.choice'.
        return NotImplementedError("Initialisation method is not implemented.")

    def _assignment(self, data : np.ndarray, centers : np.ndarray) -> np.ndarray:
        # TODO: Compute the assignment of data points to the given centres.
        # - For each data point, return the index of the closest cluster.
        #   - Do not use 'for loop' over 'data' but over 'centers' - it will be much faster.
        #   - Use 'self._distance' to compute distances. The function can compute distance from the center for all points at once.
        raise NotImplementedError("Assignment method is not implemented.")

    def _newCenters(self, data : np.ndarray, labels : np.ndarray) -> np.ndarray:
        # TODO: Compute the new cluster centers from the assignment given by 'labels'.
        # - A new center is the mean of data points belonging to the respective cluster if 'self._medoid' is False.
        # - If 'self._medoid' is True then:
        #   - In each cluster, find the point with minimal mean distance to all other points in the same cluster, and choose it as the new centre.
        #   - Distance should be computed using 'self._distance' - it can compute the distance between one point and all others at once.
        # - NOTE: If there are no points assigned to a particular cluster then select a random data point as the new center.
        raise NotImplementedError("Computation of new centres is not implemented.")

    def _wce(self, data : np.ndarray, centers : np.ndarray, labels : np.ndarray) -> float:
        # TODO: Compute the within cluster error (WCE) as the sum of distances between points and the cluster centre they belong to.
        # - This is WSS for the squared euclidean distance.
        raise NotImplementedError("WCE method is not implemented.")

    def _optimize(self, data : np.ndarray, init_centers : np.ndarray) -> Tuple[np.ndarray, float]:
        # TODO: Compute one run of the optimization.
        # - Run 'self._max_iter' iterations of Expectation-Maximization updates. In each:
        #   - Compute the cluster assignment for data with the current cluster centres. (Expectation)
        #   - Compute the new centres from the data point assignment. (Maximization)
        #   - Compute the WCE error of the iteration.
        #   - Compute the distances between cluster centres from the previous iteration and the current ones (use squared euclidean distance).
        #   - If the sum of the cluster position differences is less than 'self._tol' then break the for loop.
        #   - Print debugging information about the iteration (number, wce, change in cluster position) if 'self._verbose' is True.
        #     - e.g. print("Iteration {:>4}, WCE error: {:>7.4f}, difference from previous centers: {:>7.4f}".format(k, current_wce, difference))
        #
        # - Return the centers and WCE from the last iteration.
        # - Use 'self._assignment', 'self._newCenters' and 'self._wce' to split the code.
        raise NotImplementedError("Optimisation method is not implemented.")

    def fit(self, data : np.ndarray) -> None:
        # TODO: Train the naive K-Medoids algorithm.
        # - Run 'self._restarts' iterations of fully restarted algorithm. In each:
        #   - Print out the number of the current restart if 'self._verbose' is True.
        #   - Compute initial cluster centres.
        #   - Run one full optimization starting with the initial centres.
        #   - Remember centers and WCE for every restart.
        # - Select the smallest WCE and corresponding centres and store them in 'self.wce' and 'self.centers'
        # - Use 'self._initialize' and 'self._optimize' to split the code.
        raise NotImplementedError("Fit method is not implemented.")

    def predict(self, data : np.ndarray) -> np.ndarray:
        # TODO: Use 'self._assignment' to compute the predicted labels with respect to 'self.centers'.
        raise NotImplementedError("Predict method is not implemented.")

def testNoError(data : np.ndarray, generator : np.random.RandomState, distance : Callable[[np.ndarray, np.ndarray], np.ndarray], repeat : int = 5, clusters : int = 10) -> bool:
    """Runs the manual K-Medoids several times to find out whether it crashes due to random circumstance - empty clusters not handled correctly."""
    # NOTE: The potential exception should tell you what is the problem.
    for _ in range(repeat):
        for k in range(1, clusters):
            manualKmedoids = NaiveKMedoids(k, distance, generator=generator, verbose=False)
            manualKmedoids.fit(data)
            predictions = manualKmedoids.predict(data)
    return True

def main(args : argparse.Namespace):
    # NOTE: Naive implementation of K-Medoids/K-Means algorithm.
    generator = np.random.RandomState(args.seed)

    # Select data for testing.
    if args.data == "square":
        data = lab11_help.generateSquareClusters(args.points, args.offset, generator)
    elif args.data == "normal":
        mu1 = [-2, -4]
        sigma1 = [[1, 0], [0, 1]]
        mu2 = [0, 0]
        sigma2 = [[1, 0], [0, 1]]
        mu3 = [3, 3]
        sigma3 = [[1, 0], [0, 1]]
        mus = [mu1, mu2, mu3]
        sigmas = [sigma1, sigma2, sigma3]
        label_values = [0, 1, 2]
        data, labels = lab11_help.mvnGenerateData(generator, args.points, mus, sigmas, label_values)
    elif args.data == "mnist":
        train = lab11_help.MnistDataset("mnist_train.npz")
        data = train.imgs[: args.mnist_count]
    else:
        raise ValueError("Unknown dataset name: {}!".format(args.data))

    # TODO: Evaluate the algorithm on different number of clusters several times to find out
    # if there is an obvious mistake which causes the algorithm to crash.
    # - Try this on 'square'/'normal', do not try it on 'mnist'.
    # - It can happen if you handle empty clusters incorrectly.
    if args.test:
        print("Euclidean test passed." if testNoError(data, generator, euclidean) else "Exception!")
        print("Square euclidean test passed." if testNoError(data, generator, squareEuclidean) else "Exception!")
        print("Cosine test passed." if testNoError(data, generator, cosine) else "Exception!")
        print("Manhattan test passed." if testNoError(data, generator, manhattan) else "Exception!")
        print("Chebyshev test passed." if testNoError(data, generator, chebyshev) else "Exception!")

    # Selection of distance metrics through arguments.
    distanceMetrics = {
        "euclidean" : euclidean,
        "squareEuclidean" : squareEuclidean,
        "cosine" : cosine,
        "manhattan" : manhattan,
        "chebyshev" : chebyshev,
        "hamming" : hamming,
    }
    if args.distance not in distanceMetrics:
        raise ValueError("Unknown distance metric: {}.".format(args.distance))
    distanceMetric = distanceMetrics[args.distance]

    # TODO: Compute K-Medoids clustering on all three datasets with the available distance metrics and
    # observe the results.
    # If you set parameter 'medoid' to False, the algorithm will compute euclidean means of clusters as centers
    # so K-Means is equivalent to 'distanceMetric = euclidean/squareEuclidean', 'medoid=False'.
    # Further, 'verbose=True' enables printing of optimisation information.
    manualKmedoids = NaiveKMedoids(args.clusters, distanceMetric, args.restarts, args.max_iter, generator=generator, medoid=not args.mean_centers, verbose=True)
    manualKmedoids.fit(data)

    # Let's visually test if the algorithm is doing what it is supposed to do.
    predictions = manualKmedoids.predict(data)

    if args.data in ["square", "normal"]:
        colours = ["C{}".format(c) for c in predictions]
        _, ax = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={"aspect": "auto"})
        lab11_help.showSilhouette(data, predictions, args.clusters, ax[0])
        ax[1].set_title("Clusters with centres marked by X")
        ax[1].scatter(data[:, 0], data[:, 1], color=colours)
        ax[1].scatter(manualKmedoids.centers[:, 0], manualKmedoids.centers[:, 1], marker="x", s=200, c="black")
        plt.tight_layout()
        plt.show()
    elif args.data == "mnist":
        _, ax = plt.subplots(1, args.clusters, figsize=(17, 3), subplot_kw={"aspect": "auto"})
        for k in range(args.clusters):
            ax[k].set_title("Cluster {} centre".format(k))
            ax[k].imshow(np.reshape(manualKmedoids.centers[k], [28, 28]), cmap="Greys_r")
        plt.tight_layout()
        plt.show()

        lab11_help.showSilhouette(data, predictions, args.clusters, None)

    # TODO (Optional): If you want, you can copy the evaluation methods from previous exercises
    # and compute WSS and silhouette score graphs.

    pass


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

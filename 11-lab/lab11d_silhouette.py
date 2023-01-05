
import argparse
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import sklearn.cluster
import lab11_help


parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--seed", default=None, type=int, help="Seed for RNG.")
parser.add_argument("--data", default="square", type=str, help="Data for clustering, one of: 'square', 'normal', 'mnist'")
parser.add_argument("--method", default="single", type=str, help="Clustering method: either one of hierarchical methods or 'kmeans'.")
parser.add_argument("--clusters", default=5, type=int, help="Number of requested clusters.")
parser.add_argument("--points", default=200, type=int, help="Number of points to generate.")
parser.add_argument("--offset", default=1.1, type=float, help="Default offset of the square data.")
parser.add_argument("--mnist_count", default=500, type=int, help="Number of images taken from the MNIST dataset.")


def evalClusters(data : np.ndarray, clusterList : Sequence[int], method : str) -> np.ndarray:
    """Evaluates the silhouette score for different number of clusters using hierarchical clustering or K-means."""
    # TODO: Compute the linkage using method 'method' if it's a hierarchical method and not 'kmeans'.
    if method != "kmeans":
        linkage_matrix = None
    
    # TODO: For every "number of clusters" in 'clusterList' compute clustering with the given 'method'.
    # Then compute the silhouette score for this clustering using 'silhouette_score' method.
    # Return the list of silhouette scores - it should have the same length as 'clusterList'
    scores = np.zeros(len(clusterList))
    for i, k in enumerate(clusterList):
        if method == "kmeans":
            # TODO: Compute K-means clustering using 'sklearn.cluster.KMeans' and assign cluster predictions to the 'labels' variable.
            labels = None
        else:
            # TODO: Compute hierarchical clustering using 'fcluster' with the given 'method'.
            labels = None
        score = silhouette_score(data, labels)
        scores[i] = score
    return np.asarray(scores)

def wss(data : np.ndarray, labels : np.ndarray, clusters : int) -> float:
    """Computes the within sum of squares score for the given number of clusters."""
    # TODO: Compute WSS according to the formula from the lecture.
    sum = None
    return sum

def evalWSS(data : np.ndarray, cluster_list : Sequence[int], method : str) -> np.ndarray:
    """Evaluates the WSS score for different number of clusters using hierarchical clustering or K-means."""
    # TODO: Compute the linkage using method 'method' if it's a hierarchical method and not 'kmeans'.

    # TODO: For every "number of clusters" in 'clusterList' compute clustering with the given 'method'.
    # Then compute the WSS score for this clustering using 'wss' method.
    # Return the list of WSS scores - it should have the same length as 'clusterList'
    # - This function is effectively the same as 'evalClusters' with 'wss' instead of 'silhouette_score'.
    wss_scores = None
    return np.asarray(wss_scores)

def main(args : argparse.Namespace):
    generator = np.random.RandomState(args.seed)

    # Choose the dataset for clustering.
    if args.data == "square":
        data = lab11_help.generateSquareClusters(args.points, args.offset, generator)
    elif args.data == "normal":
        mu1 = [-4, -4]
        sigma1 = [[1, 0], [0, 1]]
        mu2 = [0, 0]
        sigma2 = [[1, 0], [0, 1]]
        mu3 = [4, 4]
        sigma3 = [[1, 0], [0, 1]]
        mus = [mu1, mu2, mu3]
        sigmas = [sigma1, sigma2, sigma3]
        label_values = [0, 1, 2]
        data, labels = lab11_help.mvnGenerateData(generator, args.points, mus, sigmas, label_values)
    elif args.data == "mnist":
        train = lab11_help.MnistDataset("mnist_train.npz")
        data = train.imgs[: args.mnist_count]
    else:
        raise ValueError("Unknown data type: '{}'.".format(args.data))
    
    if args.method == "kmeans":
        # Compute K-means clustering using 'sklearn.cluster.KMeans' and assign cluster predictions to the 'labels' variable.
        kmeans = sklearn.cluster.KMeans(args.clusters)
        predictions = kmeans.fit_predict(data)
    else:
        # Perform hierarchical clustering on the data.
        linkage_matrix = linkage(data, method=args.method)
        # Compute hierarchical clustering using 'fcluster' with the given 'method'.
        predictions = fcluster(linkage_matrix, t=args.clusters, criterion="maxclust")
        predictions -= 1

    # TODO: Compare the silhouette graph and clustering results for 4, 5 and other numbers of clusters.
    # - Use 'mnist' data only with 'kmeans' clustering - otherwise, you have to compute cluster centres from 'fcluster' result.
    if args.data in ["square", "normal"]:
        colours = ["C{}".format(c) for c in predictions]
        _, ax = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={"aspect": "auto"})
        lab11_help.showSilhouette(data, predictions, args.clusters, ax[0])
        ax[1].scatter(data[:, 0], data[:, 1], color=colours)
        ax[1].set_title("Clustering result")
        plt.tight_layout()
        plt.show()
    elif args.data == "mnist":
        _, ax = plt.subplots(1, args.clusters, figsize=(17, 3), subplot_kw={"aspect": "auto"})
        for k in range(args.clusters):
            ax[k].set_title("Cluster {} centre".format(k))
            ax[k].imshow(np.reshape(kmeans.cluster_centers[k], [28, 28]), cmap="Greys_r")
        plt.tight_layout()
        plt.show()

        lab11_help.showSilhouette(data, predictions, args.clusters, None)

    # TODO: Finish the method 'evalClusters' and evaluate silhouette scores for different number of clusters.
    # - Compare the silhouette score graphs for 'kmeans' and different linkage methods, mainly 'single', 'complete', 'average' and 'ward'.
    # - Determine the best number of clusters by maximization of the silhouette score - does it work?
    # TODO: Finish the 'evalWSS' and 'wss' methods to compute within sum of squares score.
    # - Use elbow method to determine the best number of clusters for the data - does it work?
    _, ax = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={"aspect": "auto"})
    clusterList = range(2, 21)
    scores = evalClusters(data, clusterList, args.method)
    lab11_help.plotScores(clusterList, scores, "Silhouette score for MVN data.", "Number of clusters", "Silhouette score", ax[0])
    clusterList = range(1, 21)
    wssScores = evalWSS(data, clusterList, args.method)
    lab11_help.plotScores(clusterList, wssScores, "WSS Scores for MVN data", "Number of clusters", "Total Within Sum of Squares", ax[1])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)


import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import lab11_help


parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--task", default="lecture", type=str, help="Task or example that is being evaluated, one of: 'lecture', 'square', 'normal'.")
parser.add_argument("--show_generated", default=False, action="store_true", help="Plots the generated data.")
parser.add_argument("--method", default="ward", type=str, help="Method for the computation of cluster distance.")
parser.add_argument("--seed", default=None, type=int, help="RNG seed for reproducibility.")
parser.add_argument("--points", default=200, type=int, help="Number of points to generate.")
parser.add_argument("--offset", default=1.1, type=float, help="Default offset of the square data.")
parser.add_argument("--clusters", default=10, type=int, help="Number of requested clusters.")

def lectureData() -> np.ndarray:
    """Creates example data from the lecture."""
    return np.asarray([[3.8, 3.8], [2, 2.4], [3.1, 2], [2.2, 1.1], [1, 2.8], [4, 1.8]])

def showData(data : np.ndarray, args : argparse.Namespace, labels : np.ndarray = None) -> None:
    if args.show_generated:
        _, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={"aspect": "equal"})
        if labels is not None:
            colours = np.asarray(["red", "green", "blue", "magenta", "cyan", "yellow"])
            ax.scatter(data[:, 0], data[:, 1], c=colours[labels])
        else:
            ax.scatter(data[:, 0], data[:, 1])
        plt.tight_layout()
        plt.show()

def lectureExample(generator : np.random.RandomState, args : argparse.Namespace):
    data = lectureData()
    showData(data, args)
    
    # NOTE: 'method' describes the method of determining closeness for merging.
    # 'ward' - Ward's method
    # 'average' - Average distance of points in clusters.
    # 'complete' - Maximum distance of points in clusters.
    # 'single' - Minimum distance of points in clusters.
    # 'weighted' - Weights by the distance to a third cluster.
    # 'centroid' - Distance of centroids.
    # 'median' - Same as centroid but new centroid is the average of former centroids.
    linkage_matrix = linkage(data, method=args.method)
    
    # TODO: Plot and compare the dendrograms computed using different linkage methods.
    # - Compare you results with diagrams on 65th slide of the presentation from a previous week (MLCV_9.pdf)
    _, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={"aspect": "auto"})
    dendrogram(linkage_matrix, ax=ax)
    plt.tight_layout()
    plt.show()

def squareDataExample(generator : np.random.RandomState, args : argparse.Namespace):
    # Prepare data with 4 obvious clusters.
    square_data = lab11_help.generateSquareClusters(args.points, args.offset, generator)
    showData(square_data, args)

    linkage_matrix_all = linkage(square_data, method=args.method)
    _, ax = plt.subplots(1, 1, figsize=(11, 7), subplot_kw={"aspect": "auto"})
    ax[0].set_title("Complete dendrogram")
    dendrogram(linkage_matrix_all, color_threshold="default", ax=ax[0])
    ax[1].set_title("Truncated dendrogram")
    dendrogram(linkage_matrix_all, p=args.clusters, truncate_mode="lastp", ax=ax[1])
    plt.tight_layout()
    plt.show()

def normalDataExample(generator : np.random.RandomState, args : argparse.Namespace):
    # Create MVN data.
    mu1 = [1, 1]
    sigma1 = [[0.8, 0], [0, 0.8]]
    mu2 = [3, 7]
    sigma2 = [[0.8, 0], [0, 0.8]]
    mu3 = [6, 4]
    sigma3 = [[0.8, 0], [0, 0.8]]
    mus = [mu1, mu2, mu3]
    sigmas = [sigma1, sigma2, sigma3]
    label_values = [0, 1, 2]
    data, labels = lab11_help.mvnGenerateData(generator, args.points, mus, sigmas, label_values)
    showData(data, args, labels)

    linkage_matrix = linkage(data, method=args.method)
    # 't' specifies number of requested clusters.
    predictions = fcluster(linkage_matrix, t=3, criterion="maxclust")
    predictions -= 1 # The clusters have numbers starting with 1

    # Visualisation of the true clusters and the result of hierarchical clustering.
    # NOTE: The colours might be switched around in the clustering result because the algorithm
    # has no notion of true labels.
    # TODO: Visually compare the true classes and hierarchical clusters. Are the clusters good?
    colours = np.asarray(["red", "green", "blue"])
    _, ax = plt.subplots(1, 2, figsize=(11, 7), subplot_kw={"aspect": "auto"})
    ax[0].scatter(data[:, 0], data[:, 1], c=colours[labels])
    ax[0].set_title("True classes")
    ax[1].scatter(data[:, 0], data[:, 1], c=colours[predictions])
    ax[1].set_title("Clustering result")
    plt.tight_layout()
    plt.show()

def main(args : argparse.Namespace):
    # NOTE: Hierarchical agglomerative clustering techniques.
    generator = np.random.RandomState(args.seed)

    tasks = {
        "lecture" : lectureExample,
        "square" : squareDataExample,
        "normal" : normalDataExample,
    }
    if args.task not in tasks.keys():
        raise ValueError("Unrecognised task: '{}'.".format(args.task))
    tasks[args.task](generator, args)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

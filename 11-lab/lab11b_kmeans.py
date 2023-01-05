
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import lab11_help


parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--seed", default=None, type=int, help="Seed for RNG.")
parser.add_argument("--data", default="square", type=str, help="Which data should be used: 'square', 'normal'.")
parser.add_argument("--clusters", default=5, type=int, help="Number of requested clusters.")
parser.add_argument("--points", default=200, type=int, help="Number of points to generate.")
parser.add_argument("--offset", default=1.1, type=float, help="Default offset of the square data.")

def main(args : argparse.Namespace):
    # NOTE: K-means algorithm example.

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
    else:
        raise ValueError("Unknown dataset name: {}!".format(args.data))

    # TODO: Try computing K-Means with different number of clusters on the available datasets.
    kmeans = sklearn.cluster.KMeans(args.clusters, random_state=generator)
    kmeans.fit(data)

    predictions = kmeans.predict(data)

    # Optionally, visualise the clustering result.
    colours = ["C{}".format(c) for c in predictions]
    _, ax = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={"aspect": "auto"})
    lab11_help.showSilhouette(data, predictions, args.clusters, ax[0])
    ax[1].scatter(data[:, 0], data[:, 1], color=colours)
    ax[1].set_title("Clustering result")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

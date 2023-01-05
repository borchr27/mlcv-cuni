
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from PIL import Image
import skimage.transform
import sklearn.cluster


parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--method", default="ward", type=str, help="Clustering method: either one of hierarchical methods or 'kmeans'.")
parser.add_argument("--clusters", default=10, type=int, help="Number of requested clusters.")


def main(args : argparse.Namespace):
    # NOTE: Cluster-based colourmap.

    with Image.open("slunecnice.jpg") as im_handle:
        img = np.array(im_handle, dtype=np.int32)
        # NOTE: We have to scale down the image because hierarchical algorithm is not built for pixels.
        # K-means can handle the full image but it takes a lot of time for higher number of clusters.
        # - So we scale the image in both cases.
        img = skimage.transform.resize(img, [int(img.shape[0] * 0.25), int(img.shape[1] * 0.25)])
        img = (img - np.min(img)) / np.ptp(img)

    # Let's flatten the image (channels become the 2nd dimension).
    flat_img = np.reshape(img, [-1, 3])

    if args.method == "kmeans":
        # TODO: Compute K-means clustering using 'sklearn.cluster.KMeans' with 'args.clusters' clusters.
        kmeans = None
        # TODO: Compute cluster prediction using 'predict' method of the KMeans class.
        predictions = None

        # TODO: Use computed cluster centres 'kmeans.cluster_centers_' as the colormap for the image.
        colormap = None
    else:
        # TODO: Compute hierarchical clustering 'linkage' using 'args.method' method.
        linkage_matrix = None
        # TODO: Compute cluster prediction into 'args.clusters' classes using 'fcluster'.
        # - Do not forget to subtract one from the predictions to make them zero-based.
        predictions = None

        # Create an empty colourmap.
        colormap = np.zeros([args.clusters, 3])
        # TODO: Compute median colour of every cluster and store it in 'colormap'.
        raise NotImplementedError("Missing median colour computation.")

    # Reconstruction of the image using our colourmap.
    reconstructed = colormap[predictions]
    reconstructed = np.reshape(reconstructed, img.shape)

    # Absolute error of the reconstruction.
    difference = np.abs(img - reconstructed)

    # Visualisation of the recoonstruction.
    # TODO: Visually compare the result of 2, 4, 8, 16, 32, 64, 128 and 256 clusters for both kmeans and hierarchical methods.
    _, ax = plt.subplots(3, 1, figsize=(6, 9), subplot_kw={"aspect": "auto"})
    ax[0].imshow(reconstructed)
    ax[0].set_title("Reconstruction with {} clusters by '{}' method".format(args.clusters, args.method))
    ax[1].imshow(img)
    ax[1].set_title("Original")
    ax[2].imshow(difference)
    ax[2].set_title("Difference: Abs(Original - Reconstruction)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

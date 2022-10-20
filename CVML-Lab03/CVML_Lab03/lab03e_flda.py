
import os
import argparse
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import sklearn.decomposition
import lab03_help

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--mnist_data", default="mnist_train.npz", type=str, help="Path to the MNIST dataset (images or combined).")
parser.add_argument("--mnist_labels", default=None, type=str, help="Path to the MNIST labels (if they are in a separate file).")
parser.add_argument("--limit_histograms", default=4, type=int, help="Number of classes visualised in histogram analysis tool.")

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

def main(args : argparse.Namespace):
    # TODO: Load MNIST data, which we prepared in the first practical.
    # - Change the arguments of the following class construction.
    dataset = MnistDataset(args.mnist_data, labels=args.mnist_labels)
    imgs = dataset.imgs # Change if your image data block is called differently.
    labels = dataset.labels # Change if your label data block is called differently.

    # TODO: Select and use only 1000 images and make sure that the images are in rows.

    # ===== FLDA =====
    # Get the unique labels and their count (It's probably 10 but it could be less).
    cnames, ix, ic = np.unique(labels, return_index=True, return_inverse=True)
    C = np.size(cnames)

    X = imgs
    k = X.shape[1]

    Sv = np.zeros([k, k])
    Sm = np.zeros([k, k])

    # TODO: Compute the mean of the data (assumes that images are in rows).
    mx = np.mean(X, axis=0)

    for j in cnames:
        # Get the samples for the current class (assumes that images are in rows).
        Xj = X[(labels==j), :]
        # TODO: Compute the mean of the data belonging to the class.
        # TODO: Compute the size of the class (number of images in the class).
        # TODO: Update the value of the matrix Sv.
        # TODO: Update the value of the matrix Sm.
        # - Look carefully at the shapes of the mean variables.

    # Solve the general eigenvalue problem:
    # - We use pseudo-inverse to avoid problems with singular matrices. This might produce
    #   complex eigenvalue decomposition and so we take only the real part of the result.
    #   We lose some information this way, but it is a straight-forward solution.
    M = np.linalg.pinv(Sv) @ Sm
    D, B_FLDA = scipy.linalg.eig(M)
    D = D.real
    B_FLDA = B_FLDA.real
    # Sort the eigenvectors
    descOrder = np.argsort(-D)
    D = D[descOrder]
    # The matrix produced by scipy.linalg.eig is in column-major form.
    B_FLDA = B_FLDA[:, descOrder]
    # Keep at most (c - 1) eigenvectors.
    B_FLDA = B_FLDA[:, :C-1]

    # Project the data
    X_FLDA = X @ B_FLDA

    colours = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'brown', 'coral', 'gold']
    # Plot the data in 3D (only the first three dimensions)
    _, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    ax.set_title("The first 3 PCs of FLDA")
    for i in cnames:
        digitIdxs = labels == i
        ax.scatter(
            X_FLDA[digitIdxs, 0], X_FLDA[digitIdxs, 1], X_FLDA[digitIdxs, 2],
            marker='o', color=colours[i], label="{}".format(i)
        )
    ax.legend()
    plt.tight_layout()
    plt.show()

    # TODO: Compute PCA using scikit-learn.
    pca = None

    # Plot the explained and cumulative variances.
    _, ax = plt.subplots(1, 1, subplot_kw={'aspect': 'auto'})
    ax.set_title("Explained variance")
    ax.axhline(color='black', ls='--')
    ax.axvline(color='black', ls='--')
    ax.plot(pca.explained_variance_ratio_, color='red', label="Explained")
    ax.plot(np.cumsum(pca.explained_variance_ratio_), color='blue', label="Cumulative")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Compute the new coordinates
    # Note that sklearn.decomposition.PCA.transform centres the data.
    X_PCA = X @ pca.components_.T
    #X_PCA = pca.transform(X)

    # Plot the first three PCs in 3D
    _, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    ax.set_title("The first 3 PCs of PCA")
    for i in cnames:
        digitIdxs = labels == i
        ax.scatter(
            X_PCA[digitIdxs, 0], X_PCA[digitIdxs, 1], X_PCA[digitIdxs, 2],
            marker='o', color=colours[i], label="{}".format(i)
        )
    ax.legend()
    plt.tight_layout()
    plt.show()

    # TODO: You can change the limit of the number of classes which will be shown
    # in the histograms. There is 10 in total, so try, for instance, 2, 3, 5, 10.
    # Note, that the histograms are very cluttered for more than 4 classes.
    classCountLimit = args.limit_histograms
    selectedExamples = X_PCA[labels < classCountLimit, :3]
    #selectedExamples = X_FLDA[labels < classCountLimit, :3]

    lab03_help.plotHistograms(selectedExamples, labels[labels < classCountLimit])


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
